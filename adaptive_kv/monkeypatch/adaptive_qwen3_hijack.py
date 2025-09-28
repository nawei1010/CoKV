from typing import Callable, Optional, Union

import torch
from torch import nn
from transformers.cache_utils import Cache, DynamicCache
from .utils import DynamicCacheSplitHeadFlatten, init_adaptive_snapkv

from transformers.modeling_flash_attention_utils import FlashAttentionKwargs

from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.processing_utils import Unpack
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.utils import TransformersKwargs
from transformers.models.qwen3.modeling_qwen3 import (
    apply_rotary_pos_emb,
    repeat_kv,
    eager_attention_forward
)

# Import flash attention functions
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    _flash_attn_available = True
except ImportError:
    _flash_attn_available = False


def adaptive_qwen3_Attention_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        
        # Initialize adaptive snapkv if using DynamicCacheSplitHeadFlatten
        if isinstance(past_key_value, DynamicCacheSplitHeadFlatten):
            init_adaptive_snapkv(self)
        
        bsz, q_len, _ = hidden_states.size()
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        #kv_seq_len should be initialized in each prefill step
        if q_len != 1:
            self.kv_seq_len = 0

        # Project to query, key, value
        query_states = self.q_norm(self.q_proj(hidden_states).reshape(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).reshape(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).reshape(hidden_shape).transpose(1, 2)
    
        # Get sequence length for KV
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if hasattr(self, "kv_seq_len"):  # adaptive kv cache
                if self.kv_seq_len != 0:
                    kv_seq_len += self.kv_seq_len
        cos, sin = position_embeddings
       
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
      
        # Handle KV cache
        if past_key_value is not None and isinstance(past_key_value, DynamicCacheSplitHeadFlatten):
           
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            # Check if this is the first pass (prefill) or subsequent passes (decoding)
            if key_states.shape[-2] == kv_seq_len:  # First pass - prefill phase
               
                self.kv_seq_len = kv_seq_len
                # Use kv_cluster to compress the KV cache
                
                key_states_compress, value_states_compress = self.kv_cluster.update_kv(
                    key_states, query_states, value_states, self.layer_idx
                )
            
                past_key_value.update(key_states_compress, value_states_compress, self.layer_idx, cache_kwargs)
                
                # Use standard attention for prefill
                attention_interface: Callable = eager_attention_forward
                if self.config._attn_implementation != "eager":
                    attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

                attn_output, attn_weights = attention_interface(
                    self,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    scaling=self.scaling,
                    sliding_window=self.sliding_window,
                    **kwargs,
                )
            else:  # Decoding phase - use varlen flash attention
                self.kv_seq_len += q_len
                
                # In decoding phase, we need to ensure metadata is available
                # The metadata should have been initialized in the prefill phase
                if not (hasattr(self.kv_cluster, 'head_lens') and self.kv_cluster.head_lens is not None and
                       hasattr(self.kv_cluster, 'cu_klen') and self.kv_cluster.cu_klen is not None):
                    # If metadata is not ready, this means we're in the first decoding step
                    # We need to initialize it first
                    print("Warning: Metadata not ready in decoding phase, initializing...")
                    # This should not happen in normal operation, but let's handle it gracefully
                    pass
                
                # Add metadata for DynamicCacheSplitHeadFlatten
                cache_kwargs["head_lens"] = self.kv_cluster.head_lens
                cache_kwargs["cu_klen"] = self.kv_cluster.cu_klen
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
                # Update metadata (only if initialized)
                if hasattr(self.kv_cluster, 'klen_sum') and self.kv_cluster.klen_sum is not None:
                    self.kv_cluster.klen_sum += self.config.num_attention_heads
                if hasattr(self.kv_cluster, 'max_seqlen_k') and self.kv_cluster.max_seqlen_k is not None:
                    self.kv_cluster.max_seqlen_k += 1
                if hasattr(self.kv_cluster, 'cu_klen') and hasattr(self.kv_cluster, 'cu_offset') and \
                   self.kv_cluster.cu_klen is not None and self.kv_cluster.cu_offset is not None:
                    self.kv_cluster.cu_klen += self.kv_cluster.cu_offset
                if hasattr(self.kv_cluster, 'head_lens') and self.kv_cluster.head_lens is not None:
                    self.kv_cluster.head_lens += 1
                
                # Use flash_attn_varlen_func for decoding with adaptive cache
                if (_flash_attn_available and self.config._attn_implementation == "flash_attention_2"):

                    # Reshape for varlen flash attention
                    query_states = query_states.reshape(-1, self.num_key_value_groups, self.head_dim)
                    key_states = key_states.reshape(-1, 1, self.head_dim)
                    value_states = value_states.reshape(-1, 1, self.head_dim)
                    
                    # Check if metadata is available, if not use fallback
                    if (hasattr(self.kv_cluster, 'cu_qlen') and self.kv_cluster.cu_qlen is not None and
                        hasattr(self.kv_cluster, 'cu_klen') and self.kv_cluster.cu_klen is not None and
                        hasattr(self.kv_cluster, 'max_seqlen_k') and self.kv_cluster.max_seqlen_k is not None):
                        cu_seqlens_q = self.kv_cluster.cu_qlen
                        cu_seqlens_k = self.kv_cluster.cu_klen
                        max_seqlen_q = 1
                        max_seqlen_k = self.kv_cluster.max_seqlen_k
                    else:
                        # Metadata not ready yet, fall back to eager attention
                        raise RuntimeError(
                            "Flash attention metadata not ready. This should not happen in normal operation."
                        )
                    
                    attn_output = flash_attn_varlen_func(
                        query_states, key_states, value_states, 
                        cu_seqlens_q, cu_seqlens_k, 
                        max_seqlen_q, max_seqlen_k, 
                        causal=True
                    )
                  
                    attn_output = attn_output.reshape(bsz, self.config.num_attention_heads, q_len, self.head_dim)


                    # Reorder to (bsz, q_len, num_heads * head_dim) without relying on config.hidden_size
                    # attn_output = attn_output.transpose(1, 2).contiguous()
                    # attn_output = attn_output.reshape(bsz, q_len, attn_output.size(2) * attn_output.size(3))

                    attn_output = attn_output.transpose(0, 1).contiguous()
                    attn_output = attn_output.reshape(bsz, q_len, attn_output.size(0) * attn_output.size(3))
                    attn_weights = None
                else:
                    # When adaptive cache returns flattened format but flash attention is not available,
                    # we need to handle this case differently
                    if len(key_states.shape) == 2:
                        # Flattened format from adaptive cache - flash attention is required
                        # Fall back to a simple attention implementation or raise an error
                        raise RuntimeError(
                            "Adaptive KV cache with flattened format requires Flash Attention 2. "
                            "Please install flash-attn or disable adaptive cache."
                        )
                    else:
                        # Standard 4D format - can use eager attention
                        attention_interface: Callable = eager_attention_forward
                        attn_output, attn_weights = attention_interface(
                            self,
                            query_states,
                            key_states,
                            value_states,
                            attention_mask,
                            dropout=0.0 if not self.training else self.attention_dropout,
                            scaling=self.scaling,
                            sliding_window=self.sliding_window,
                            **kwargs,
                        )

            
                    
        else:
            # Standard cache handling for non-adaptive cache
            if past_key_value is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

            # Use standard attention
            attention_interface: Callable = eager_attention_forward
            if self.config._attn_implementation != "eager":
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                sliding_window=self.sliding_window,
                **kwargs,
            )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
    
    
def adaptive_qwen3_Model_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # if use_cache and past_key_values is None:
        if use_cache:
           past_key_values = DynamicCacheSplitHeadFlatten.from_legacy_cache(past_key_values)
     

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )
        
def quant_and_dequant(key_or_value_states: torch.Tensor, head_bits) -> torch.Tensor:
    """
    Simulate quantization + dequantization for each key/value head group with specified bits
    (per token, per head group, dynamic range computed along head_dim).

    Args:
        key_or_value_states: (batch_size, num_key_value_groups, seqlen, head_dim)
        head_bits: list[int] or 1D Tensor of length num_key_value_groups, specifying quantization bits for each head group.
                   Convention:
                     - bit <= 0: No quantization (return original values)
                     - bit == 1: 1-bit sign quantization (-1/+1), (0 possible via scaling and rounding)
                     - 1 < bit < 16: Symmetric uniform quantization to [-qmax, qmax], qmax = 2^(bit-1)-1
                     - bit >= 16: No quantization (keep as float)
    Returns:
        dequant_states: Tensor of the same shape, after quantization and dequantization
    Note:
        Quantization scale is computed per (batch, group, token) (amax over head_dim),
        which reduces error and adapts to incremental length. To share scale over the whole sequence,
        change to amax over dim=(-2,-1).
    """

    if head_bits is None:
        return key_or_value_states

    x = key_or_value_states
    if not torch.is_tensor(x):
        raise TypeError("key_or_value_states must be a Tensor")

    bsz, G, seqlen, hd = x.shape
    device = x.device
    dtype = x.dtype

    head_bits = torch.as_tensor(head_bits, device=device, dtype=torch.int32)
    assert head_bits.numel() == G, f"head_bits length {head_bits.numel()} does not match num_key_value_groups {G}"

    # Expand bits to broadcastable shape (1, G, 1, 1)
    bits = head_bits.view(1, G, 1, 1)

    # Mask
    skip_mask = (bits >= 16) | (bits <= 0)  # Groups not quantized
    active_mask = ~skip_mask

    if active_mask.sum() == 0:
        return x  # All groups skipped

    # qmax (float, for scale calculation). Note: bit==1, qmax = 1 (range [-1,1])
    # For skipped groups, set to 1 to avoid division
    qmax = torch.where(
        active_mask,
        torch.where(bits > 1, (2 ** (bits - 1) - 1).to(device=device, dtype=dtype), torch.ones_like(bits, dtype=dtype)),
        torch.ones_like(bits, dtype=dtype),
    )

    # Calculate max absolute value per (b, g, t) -> (b, G, seqlen, 1)
    max_abs = x.abs().amax(dim=-1, keepdim=True)  # Prevent all zeros
    # For skipped groups, force max_abs=1 to avoid meaningless scale values
    max_abs = torch.where(skip_mask, torch.ones_like(max_abs), max_abs)

    # scale = max_abs / qmax ; inv_scale = qmax / max_abs
    # Avoid division by zero
    safe_max_abs = torch.where(max_abs == 0, torch.ones_like(max_abs), max_abs)
    inv_scale = qmax / safe_max_abs  # (1,G,1,1) and (b,G,t,1) broadcast => (b,G,t,1)
    # Only valid for active_mask, other groups inv_scale=1

    # Quantization (simulation): round(x * inv_scale) and clamp to [-qmax, qmax]
    x_scaled = x * inv_scale
    qmin = -qmax
    qmax_clamp = qmax
    q = torch.round(x_scaled)
    q = torch.minimum(torch.maximum(q, qmin), qmax_clamp)

    # Dequantization
    dequant = q / inv_scale

    # For skipped groups, directly restore original values
    if skip_mask.any():
        dequant = torch.where(skip_mask, x, dequant)

    return dequant


def adaptive_quantization_qwen3_Attention_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        
        key_states = quant_and_dequant(key_states, self.config.head_bits[self.layer_idx])
        value_states = quant_and_dequant(value_states, self.config.head_bits[self.layer_idx])

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,  # diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights



