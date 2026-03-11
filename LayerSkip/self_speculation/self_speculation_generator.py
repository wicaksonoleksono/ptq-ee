# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import time
from typing import List, Optional, Tuple, Dict

import colorama
import torch

import transformers
from self_speculation.generator_base import (
    GenerationConfig,
    GenerationStrategy,
    GenerationStrategyResult,
)
from self_speculation.speculative_streamer import SpeculativeTextStreamer
from self_speculation.llama_model_utils import (
    crop_past_key_values,
    decode_next_token,
    forward_early,
    forward_remainder,
)

def max_fn(x, eps=1e-6):
    x_max = torch.where(x > 0, x, 0)
    return x_max / (torch.sum(x_max) + eps)

class SelfSpeculativeGenerationStrategy(GenerationStrategy):
    def generate_token_ids(
        self,
        model: transformers.LlamaForCausalLM,
        input_ids: List[int],
        eos_token_ids: List[int],
        generation_config: GenerationConfig,
        logits_processors: Optional[transformers.generation.logits_process.LogitsProcessorList] = None,
        stopping_criteria: Optional[transformers.StoppingCriteriaList] = None,
        streamer: Optional[transformers.TextStreamer] = None,
    ) -> GenerationStrategyResult:
        # Reset cache for every new prompt
        past_key_values = None
        torch.cuda.empty_cache()

        input_ids_list = input_ids
        input_ids: torch.Tensor = torch.tensor([input_ids_list]).to(model.device)
        num_prefill_tokens = input_ids.shape[1]
        output_ids: List[int] = []
        
        # GRANULAR AUDIT LOG
        # This will store { "token": id, "layer": exit_layer, "origin": str, "timestamp": float }
        token_audit_log: List[Dict] = []
        
        acceptance_rates_list: List[float] = [] 
        
        num_layers = model.config.num_hidden_layers
        calls: int = 0
        total_draft_matches = 0
        total_generations = 0
        
        prefill_time = 0.0
        decode_start_time = 0.0
        
        while len(output_ids) < generation_config.max_steps:
            step_start = time.time()
            (
                input_ids,
                output_ids_step, # New tokens in this step
                past_key_values,
                number_of_matches,
                num_speculations,
            ) = self.single_step_speculation(
                model=model,
                input_ids_list=input_ids_list,
                input_ids=input_ids,
                output_ids=[], 
                num_speculations=min(
                    generation_config.num_speculations,
                    generation_config.max_steps - len(output_ids) - 1,
                ),
                past_key_values=past_key_values,
                exit_layer=generation_config.exit_layer,
                eos_token_ids=eos_token_ids,
                calls=calls,
                sample=generation_config.sample,
                temperature=generation_config.temperature,
                top_k=generation_config.top_k,
                top_p=generation_config.top_p,
                logits_processors=logits_processors,
                stopping_criteria=stopping_criteria,
                streamer=streamer,
            )
            
            # Record per-token origins and layers into the audit log
            step_timestamp = time.time()
            for idx, t_id in enumerate(output_ids_step):
                is_draft = idx < number_of_matches
                token_audit_log.append({
                    "token_id": int(t_id),
                    "exit_layer": int(generation_config.exit_layer if is_draft else num_layers),
                    "origin": "draft" if is_draft else "verification",
                    "timestamp": step_timestamp
                })

            if num_speculations > 0:
                acceptance_rates_list.append(float(number_of_matches) / num_speculations)
            
            output_ids.extend(output_ids_step)
            
            if calls == 0:
                prefill_time = time.time() - step_start
                decode_start_time = time.time()
                
            calls += 1
            total_draft_matches += number_of_matches
            total_generations += num_speculations
            
            # EOS Handling
            eos_found = False
            for eos_token_id in eos_token_ids:
                if eos_token_id in output_ids:
                    idx = output_ids.index(eos_token_id)
                    output_ids = output_ids[:idx]
                    token_audit_log = token_audit_log[:idx]
                    eos_found = True
                    break
            if eos_found: break
            if stopping_criteria and stopping_criteria(input_ids, None): break

        return GenerationStrategyResult(
            predicted_tokens=output_ids,
            acceptance_rate=float(total_draft_matches) / total_generations if total_generations > 0 else 0.0,
            acceptance_rates=acceptance_rates_list,
            exit_layers=[log["exit_layer"] for log in token_audit_log],
            token_origins=[1 if log["origin"]=="draft" else 0 for log in token_audit_log],
            prefill_time=prefill_time,
            decode_time=time.time() - decode_start_time if decode_start_time > 0 else 0.0,
            num_prefill_tokens=num_prefill_tokens,
            # Pass the full audit log back if needed (or we can reconstruct it from exit_layers)
        )

    def single_step_speculation(
        self,
        model: transformers.LlamaForCausalLM,
        input_ids: torch.Tensor,
        input_ids_list: List[int],
        output_ids: List[int],
        num_speculations: int,
        past_key_values: Optional[transformers.cache_utils.DynamicCache],
        eos_token_ids: List[int],
        calls: int,
        exit_layer: int,
        sample: Optional[bool] = False,
        temperature: Optional[float] = 0.7,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.95,
        logits_processors: Optional[transformers.generation.logits_process.LogitsProcessorList] = None,
        stopping_criteria: Optional[transformers.StoppingCriteriaList] = None,
        streamer: Optional[transformers.TextStreamer] = None
    ):
        prompt_length: int = input_ids.size(1)
        draft_input_ids = input_ids.clone()
        draft_output_ids: List[int] = []
        if sample:
            draft_probabilities: List[torch.Tensor] = []
        exit_query_cache = None
        
        for _ in range(num_speculations):
            draft_result = forward_early(model, draft_input_ids, past_key_values, exit_layer, exit_query_cache)
            past_key_values = draft_result.past_key_values
            exit_query_cache = draft_result.exit_query_cache
            draft_logits = draft_result.logits
            if logits_processors:
                draft_logits = logits_processors(draft_input_ids, draft_logits)
            draft_next_token, draft_next_prob = decode_next_token(logits=draft_logits, token_idx=-1, sample=sample, temperature=temperature, top_k=top_k, top_p=top_p)
            draft_next_token = draft_next_token.item()
            draft_output_ids.append(draft_next_token)
            if sample:
                draft_probabilities.append(draft_next_prob)
            draft_input_ids = torch.tensor([[draft_next_token]]).to(draft_input_ids)
            if draft_next_token in eos_token_ids: break

        draft_output_ids_tensor = torch.tensor(draft_output_ids).unsqueeze(0).to(input_ids)
        prefill_token_ids = torch.cat([input_ids, draft_output_ids_tensor], dim=-1)

        verify_results = forward_remainder(model, prefill_token_ids.int(), past_key_values, exit_layer, exit_query_cache)
        logits = verify_results.logits
        if logits_processors:
            logits = logits_processors(prefill_token_ids, logits)
        past_key_values = verify_results.past_key_values
        
        verification_logits = logits[:, prompt_length - 1 :, :]
        verified_tokens, verified_probabilities = decode_next_token(logits=verification_logits, sample=sample, temperature=temperature, top_k=top_k, top_p=top_p)

        if not sample:
            number_of_matches = 0
            for i in range(draft_output_ids_tensor.numel()):
                if draft_output_ids[i] == verified_tokens[0, i]:
                    number_of_matches += 1
                else: break
        else:
            number_of_matches = 0
            rand = torch.rand_like(draft_output_ids_tensor, dtype=torch.float)
            for i in range(draft_output_ids_tensor.numel()):
                if rand[0, i] < min(1, verified_probabilities[i, draft_output_ids[i]].item() / draft_probabilities[i][0, draft_output_ids[i]].item()):
                    number_of_matches += 1
                else:
                    verified_tokens[0][number_of_matches] = torch.multinomial(max_fn((verified_probabilities[i, :] - draft_probabilities[i])), num_samples=1).item()
                    break

        new_output_ids = draft_output_ids[:number_of_matches]
        new_output_ids.append(verified_tokens[0][number_of_matches].item())
        
        input_ids = verified_tokens[:, number_of_matches : number_of_matches + 1]
        
        past_key_values = crop_past_key_values(
            past_key_values, len(input_ids_list) + len(output_ids) + len(new_output_ids) - 1
        )

        return (input_ids, new_output_ids, past_key_values, number_of_matches, draft_output_ids_tensor.numel())
