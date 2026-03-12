# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import time
from typing import List, Optional, Tuple, Dict

import torch
import transformers
from self_speculation.generator_base import (
    GenerationConfig,
    GenerationStrategy,
    GenerationStrategyResult,
)
from self_speculation.llama_model_utils import (
    crop_past_key_values,
    decode_next_token,
    _prepare_decoder_attention_mask,
)

class AdaptiveEarlyExitStrategy(GenerationStrategy):
    """
    Implements Adaptive Early Exit (similar to EE-LLM logic).
    Exits as soon as the confidence (max prob) at an intermediate layer 
    exceeds a threshold.
    """
    def __init__(self, confidence_threshold: float = 0.9):
        self.confidence_threshold = confidence_threshold

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
        
        device = model.device
        past_key_values = transformers.cache_utils.DynamicCache()
        torch.cuda.empty_cache()

        input_ids_tensor: torch.Tensor = torch.tensor([input_ids]).to(device)
        num_prefill_tokens = input_ids_tensor.shape[1]
        output_ids: List[int] = []
        
        # Metrics
        exit_layers_list: List[int] = []
        token_origins_list: List[int] = [] # 1=Early, 0=Full (reached last layer)
        
        num_layers = model.config.num_hidden_layers
        
        prefill_time = 0.0
        decode_start_time = time.time()
        
        curr_input_ids = input_ids_tensor
        
        while len(output_ids) < generation_config.max_steps:
            batch_size, seq_length = curr_input_ids.shape
            past_len = past_key_values.get_seq_length()
            
            # 1. Prepare Embeddings and Mask
            inputs_embeds = model.model.embed_tokens(curr_input_ids)
            position_ids = torch.arange(past_len, past_len + seq_length, dtype=torch.long, device=device).unsqueeze(0)
            
            attention_mask = curr_input_ids.new_ones((batch_size, past_len + seq_length), dtype=torch.bool)
            attention_mask = _prepare_decoder_attention_mask(
                model, attention_mask, (batch_size, seq_length), inputs_embeds, past_len
            )
            
            # 2. Layer-by-Layer Loop with Adaptive Exit
            hidden_states = inputs_embeds
            actual_exit_layer = num_layers
            final_logits = None
            
            # Skip some layers for efficiency if desired, but here we probe every layer
            # after a certain point (e.g., after 50% of the model)
            min_layers = num_layers // 4 
            
            for idx, decoder_layer in enumerate(model.model.layers):
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    cache_position=torch.arange(past_len, past_len + seq_length, device=device),
                )
                hidden_states = layer_outputs[0]
                
                # Check for Early Exit
                if idx >= min_layers and idx < (num_layers - 1):
                    # Probe the LM head
                    temp_hidden = model.model.norm(hidden_states)
                    logits = model.lm_head(temp_hidden[:, -1, :]) # Check only last token
                    
                    probs = torch.softmax(logits, dim=-1)
                    max_prob, _ = torch.max(probs, dim=-1)
                    
                    if max_prob.item() >= self.confidence_threshold:
                        actual_exit_layer = idx + 1
                        final_logits = logits.unsqueeze(1) # [1, 1, V]
                        # CROP CACHE: We exited early, so we need to ensure KV cache 
                        # for FUTURE layers doesn't have junk if they were somehow updated
                        break
            
            # 3. Finalize Step
            if final_logits is None:
                # Reached the end
                actual_exit_layer = num_layers
                hidden_states = model.model.norm(hidden_states)
                final_logits = model.lm_head(hidden_states)

            if logits_processors:
                final_logits = logits_processors(curr_input_ids, final_logits)
            
            next_token, _ = decode_next_token(logits=final_logits, token_idx=-1, sample=generation_config.sample)
            token_id = next_token.item()
            
            # Record Audit
            output_ids.append(token_id)
            exit_layers_list.append(actual_exit_layer)
            token_origins_list.append(1 if actual_exit_layer < num_layers else 0)
            
            if streamer: streamer.put(next_token)
            
            # Prepare for next token
            curr_input_ids = next_token
            
            if token_id in eos_token_ids: break
            if stopping_criteria and stopping_criteria(curr_input_ids, None): break

        return GenerationStrategyResult(
            predicted_tokens=output_ids,
            exit_layers=exit_layers_list,
            token_origins=token_origins_list,
            decode_time=time.time() - decode_start_time,
            num_prefill_tokens=num_prefill_tokens,
        )
