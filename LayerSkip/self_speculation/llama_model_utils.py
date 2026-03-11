# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import transformers


@dataclass
class ForwardResult:
    logits: torch.Tensor
    past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]]
    exit_query_cache: Optional[List[torch.Tensor]] = None


# Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
def _prepare_decoder_attention_mask(
    model, attention_mask, input_shape, inputs_embeds, past_key_values_length
):
    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    combined_attention_mask = None
    if input_shape[-1] > 1:
        combined_attention_mask = _make_causal_mask(
            input_shape,
            inputs_embeds.dtype,
            device=inputs_embeds.device,
            past_key_values_length=past_key_values_length,
        )

    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = _expand_mask(
            attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
        ).to(inputs_embeds.device)
        combined_attention_mask = (
            expanded_attn_mask
            if combined_attention_mask is None
            else expanded_attn_mask + combined_attention_mask
        )

    return combined_attention_mask


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat(
            [
                torch.zeros(
                    tgt_len, past_key_values_length, dtype=dtype, device=device
                ),
                mask,
            ],
            dim=-1,
        )
    return mask[None, None, :, :].expand(
        bsz, 1, tgt_len, tgt_len + past_key_values_length
    )


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(
        inverted_mask.to(torch.bool), torch.finfo(dtype).min
    )


def top_k_top_p_filtering(
    logits: torch.FloatTensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> torch.FloatTensor:
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering

    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        top_k (`int`, *optional*, defaults to 0):
            If > 0, only keep the top k tokens with highest probability (top-k filtering)
        top_p (`float`, *optional*, defaults to 1.0):
            If < 1.0, only keep the top tokens with cumulative probability >= top_p (nucleus filtering). Nucleus
            filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimumber of tokens we keep per batch example in the output.

    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        logits = transformers.generation.logits_process.TopKLogitsWarper(
            top_k=top_k,
            filter_value=filter_value,
            min_tokens_to_keep=min_tokens_to_keep,
        )(None, logits)

    if 0 <= top_p <= 1.0:
        logits = transformers.generation.logits_process.TopPLogitsWarper(
            top_p=top_p,
            filter_value=filter_value,
            min_tokens_to_keep=min_tokens_to_keep,
        )(None, logits)

    return logits


def decode_next_token(
    logits: torch.Tensor,
    token_idx: int = None,
    sample: Optional[bool] = False,
    temperature: Optional[float] = 0.7,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = 0.95,
) -> torch.Tensor:
    if token_idx:
        logits = logits[:, -1, :]

    if not sample:
        next_token = logits.argmax(dim=-1)
        return next_token, None
    else:
        if not token_idx:
            logits.squeeze_(dim=0)
        filtered_logits = top_k_top_p_filtering(
            logits / temperature, top_k=top_k, top_p=top_p
        )
        probabilities = torch.nn.functional.softmax(filtered_logits, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1)
        if not token_idx:
            next_token.transpose_(1, 0)
        return next_token, probabilities


def crop_past_key_values(
    past_key_values: List[Tuple[torch.Tensor, torch.Tensor]],
    maximum_length: int,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    new_past: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for idx in range(len(past_key_values)):
        if (
            past_key_values[idx] is None
            or past_key_values[idx][0] == []
            or past_key_values[idx][0] is None
        ):
            break
        new_past.append(
            (
                past_key_values[idx][0][:, :, :maximum_length, :],
                past_key_values[idx][1][:, :, :maximum_length, :],
            )
        )
    past_key_values = tuple(new_past)
    return past_key_values


# Our forward_early(...) and forward_remainder(...) functions currently use transformers library's legacy KV cache implementation that is less efficient.
# To ensure an apples to apples comparison, we created this forward function to use in autoregressive decoding to ensure it uses the same KV cache implementation instead.
# FIXME: update forward_early(...) and forward_remainder(...) to use the updated more efficient KV cache implementation.
def forward(
    model: transformers.LlamaForCausalLM,
    input_ids: torch.Tensor,
    past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
) -> ForwardResult:
    device = input_ids.device
    batch_size, seq_length = input_ids.shape

    past_key_values = transformers.cache_utils.DynamicCache.from_legacy_cache(
        past_key_values
    )
    past_key_values_length = past_key_values.get_seq_length()
    seq_length_with_past = seq_length + past_key_values_length

    position_ids = torch.arange(
        past_key_values_length,
        seq_length + past_key_values_length,
        dtype=torch.long,
        device=device,
    )
    position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    cache_position = torch.arange(
        past_key_values_length, past_key_values_length + seq_length, device=device
    )

    attention_mask = input_ids.new_ones(
        (batch_size, seq_length_with_past),
        dtype=torch.bool,
    )
    inputs_embeds = model.model.embed_tokens(input_ids)
    attention_mask = _prepare_decoder_attention_mask(
        model,
        attention_mask,
        (batch_size, seq_length),
        inputs_embeds,
        past_key_values_length,
    )

    hidden_states = inputs_embeds
    position_embeddings = model.model.rotary_emb(hidden_states, position_ids)
    for decoder_layer in model.model.layers:
        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = (
            layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs
        )

    past_key_values_legacy = past_key_values.to_legacy_cache()
    hidden_states = model.model.norm(hidden_states)
    logits = model.lm_head(hidden_states)

    return ForwardResult(logits=logits, past_key_values=past_key_values_legacy)


# TODO: update forward_early(...) to use transformers' new KV cache implementation rather than legacy.
def forward_early(
    model: transformers.LlamaForCausalLM,
    input_ids: torch.Tensor,
    past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
    exit_layer: int,
    exit_query_cache: Optional[List[torch.Tensor]],
) -> ForwardResult:
    device = input_ids.device
    batch_size, seq_length = input_ids.shape

    past_key_values = transformers.cache_utils.DynamicCache.from_legacy_cache(
        past_key_values
    )
    past_key_values_length = past_key_values.get_seq_length()
    seq_length_with_past = seq_length + past_key_values_length

    position_ids = torch.arange(
        past_key_values_length,
        seq_length + past_key_values_length,
        dtype=torch.long,
        device=device,
    )
    position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    cache_position = torch.arange(
        past_key_values_length, past_key_values_length + seq_length, device=device
    )

    attention_mask = input_ids.new_ones(
        (batch_size, seq_length_with_past),
        dtype=torch.bool,
    )
    inputs_embeds = model.model.embed_tokens(input_ids)
    attention_mask = _prepare_decoder_attention_mask(
        model,
        attention_mask,
        (batch_size, seq_length),
        inputs_embeds,
        past_key_values_length,
    )

    hidden_states = inputs_embeds
    position_embeddings = model.model.rotary_emb(hidden_states, position_ids)
    for decoder_layer in model.model.layers[:exit_layer]:
        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = (
            layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs
        )

    past_key_values_legacy = past_key_values.to_legacy_cache()

    # next_cache = next_decoder_cache
    if exit_query_cache is None:
        exit_query_cache = hidden_states
    else:
        exit_query_cache = torch.cat([exit_query_cache, hidden_states], dim=1)

    hidden_states = model.model.norm(hidden_states)

    logits = model.lm_head(hidden_states)
    return ForwardResult(
        logits=logits,
        past_key_values=past_key_values_legacy,
        exit_query_cache=exit_query_cache,
    )


def forward_remainder(
    model: transformers.LlamaForCausalLM,
    input_ids: torch.Tensor,
    past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
    exit_layer: int,
    exit_query_cache: Optional[torch.Tensor],
) -> ForwardResult:
    device = input_ids.device
    batch_size, seq_length = input_ids.shape

    past_key_values = transformers.cache_utils.DynamicCache.from_legacy_cache(
        past_key_values
    )

    # Early layers (0..exit_layer-1) have cache from forward_early
    early_past_length = past_key_values.get_seq_length()  # uses layer 0

    # Late layers (exit_layer..end) may have a DIFFERENT cache length (or 0).
    # get_seq_length(layer_idx) returns 0 if that layer has no cache entries.
    late_past_length = past_key_values.get_seq_length(exit_layer)

    # New tokens that early layers haven't cached yet.
    # We must account for the fact that input_ids starts at late_past_length.
    num_tokens_to_generate = max(0, (late_past_length + seq_length) - early_past_length)

    inputs_embeds = model.model.embed_tokens(input_ids)

    # ----- Early layers setup (only if there are new tokens) -----
    if num_tokens_to_generate > 0:
        early_position_ids = torch.arange(
            early_past_length,
            early_past_length + num_tokens_to_generate,
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)
        early_cache_position = torch.arange(
            early_past_length,
            early_past_length + num_tokens_to_generate,
            device=device,
        )
        early_attn_2d = input_ids.new_ones(
            (batch_size, early_past_length + num_tokens_to_generate),
            dtype=torch.bool,
        )
        early_attention_mask = _prepare_decoder_attention_mask(
            model,
            early_attn_2d,
            (batch_size, num_tokens_to_generate),
            inputs_embeds[:, -num_tokens_to_generate:],
            early_past_length,
        )
        early_position_embeddings = model.model.rotary_emb(
            inputs_embeds[:, -num_tokens_to_generate:],
            early_position_ids,
        )

    # ----- Late layers setup -----
    # Late layers process full_hidden_states (= exit_query_cache) with their own past
    late_input_length = seq_length
    late_position_ids = torch.arange(
        late_past_length,
        late_past_length + late_input_length,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)
    late_cache_position = torch.arange(
        late_past_length,
        late_past_length + late_input_length,
        device=device,
    )
    late_attn_2d = input_ids.new_ones(
        (batch_size, late_past_length + late_input_length),
        dtype=torch.bool,
    )
    late_attention_mask = _prepare_decoder_attention_mask(
        model,
        late_attn_2d,
        (batch_size, late_input_length),
        inputs_embeds,
        late_past_length,
    )

    hidden_states = inputs_embeds
    full_hidden_states: Optional[torch.FloatTensor] = None
    late_position_embeddings = None  # computed once at first late layer

    for idx, decoder_layer in enumerate(model.model.layers):
        is_early = idx < exit_layer
        if is_early:
            if num_tokens_to_generate > 0:
                early_hidden = hidden_states[:, -num_tokens_to_generate:]
                layer_outputs = decoder_layer(
                    early_hidden,
                    attention_mask=early_attention_mask,
                    position_ids=early_position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    cache_position=early_cache_position,
                    position_embeddings=early_position_embeddings,
                )
                hidden_states = (
                    layer_outputs[0]
                    if isinstance(layer_outputs, tuple)
                    else layer_outputs
                )
            # else: early layers already cached everything, skip
        else:
            # Build full_hidden_states once at the first late layer
            if full_hidden_states is None:
                if exit_query_cache is not None:
                    if num_tokens_to_generate > 0:
                        full_hidden_states = torch.cat(
                            [
                                exit_query_cache,
                                hidden_states[:, -num_tokens_to_generate:],
                            ],
                            dim=1,
                        )
                    else:
                        # FIX: -0 slice returns entire tensor in Python, not empty!
                        full_hidden_states = exit_query_cache
                else:
                    full_hidden_states = hidden_states

                late_position_embeddings = model.model.rotary_emb(
                    full_hidden_states,
                    late_position_ids,
                )

            layer_outputs = decoder_layer(
                full_hidden_states,
                attention_mask=late_attention_mask,
                position_ids=late_position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                cache_position=late_cache_position,
                position_embeddings=late_position_embeddings,
            )
            full_hidden_states = (
                layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs
            )

    past_key_values_legacy = past_key_values.to_legacy_cache()
    final_hidden = (
        full_hidden_states if full_hidden_states is not None else hidden_states
    )
    final_hidden = model.model.norm(final_hidden)
    logits = model.lm_head(final_hidden)

    return ForwardResult(
        logits=logits,
        past_key_values=past_key_values_legacy,
        exit_query_cache=exit_query_cache,
    )
