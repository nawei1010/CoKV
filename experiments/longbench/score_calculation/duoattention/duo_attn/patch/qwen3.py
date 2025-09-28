import torch
from torch import nn
from typing import Optional
import types

from flash_attn import flash_attn_func

try:
    from transformers.models.qwen3.modeling_qwen3 import (
        Qwen3ForCausalLM,
        Qwen3Model,
        apply_rotary_pos_emb,
        repeat_kv,
    )
except Exception as e:  # pragma: no cover
    raise ImportError("transformers>=latest with qwen3 required: {}".format(e))


# ------------------------- Forward (two-way) ------------------------- #
@torch.no_grad()
def _compute_full_attn(self, hidden_states, position_embeddings, attention_mask):
    """Teacher full attention forward (no grad)."""
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    # Q/K/V + norms follow HF implementation
    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # expand kv heads
    key_states_full = repeat_kv(key_states, self.num_key_value_groups)
    value_states_full = repeat_kv(value_states, self.num_key_value_groups)

    # flash_attn expects (B, S, H, D)
    query_states_fa = query_states.transpose(1, 2)  # (B, S, H, D)
    key_states_fa = key_states_full.transpose(1, 2)
    value_states_fa = value_states_full.transpose(1, 2)

    attn_output = flash_attn_func(
        query_states_fa,
        key_states_fa,
        value_states_fa,
        causal=True,
        dropout_p=0.0,
    )  # (B, S, H, D)

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output


def qwen3_duo_attention_forward_two_way(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
    past_key_value=None,
    cache_position=None,
    **kwargs,
):
    """
    Two-way forward producing (teacher full attn, student gated streaming) concatenated on batch dim.
    hidden_states is duplicated upstream in train.py, first half = full, second half = streaming variant.
    We only learn self.full_attention_heads (size num_key_value_heads).
    """
    bsz_x2, seq_len, _ = hidden_states.shape
    assert bsz_x2 % 2 == 0
    bsz = bsz_x2 // 2

    full_hidden = hidden_states[:bsz]
    streaming_hidden = hidden_states[bsz:]

    # Teacher branch (no grad) ------------------
    with torch.no_grad():
        full_attn_output = _compute_full_attn(self, full_hidden, position_embeddings, attention_mask)

    # Student branch (needs grad w.r.t gating param only) --------------
    input_shape = streaming_hidden.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    query_states = self.q_norm(self.q_proj(streaming_hidden).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(streaming_hidden).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(streaming_hidden).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # Split full vs streaming heads according to binary mask produced by sigmoid(clamped) like behavior
    # We keep the same number of kv heads; gating later mixes outputs.
    key_states_full = repeat_kv(key_states, self.num_key_value_groups)
    value_states_full = repeat_kv(value_states, self.num_key_value_groups)

    query_states_fa = query_states.transpose(1, 2)
    key_states_fa = key_states_full.transpose(1, 2)
    value_states_fa = value_states_full.transpose(1, 2)

    streaming_attn_output = flash_attn_func(
        query_states_fa,
        key_states_fa,
        value_states_fa,
        causal=True,
        dropout_p=0.0,
    )  # (B,S,H,D)

    streaming_attn_output = streaming_attn_output.reshape(*input_shape, -1).contiguous()
    streaming_attn_output = self.o_proj(streaming_attn_output)

    # Mix per key/value head groups. Need mask expanded to query heads.
    # Derive number of query heads from tensor shape (B,H,S,D) where H = query heads
    q_heads = query_states.shape[1]
    kv_groups = self.num_key_value_groups  # available in Qwen3Attention
    q_per_kv = q_heads // kv_groups
    gate = self.full_attention_heads.clamp(0, 1)  # (kv_groups,)
    # Repeat each gate value for its query heads
    gate_expanded = gate.repeat_interleave(q_per_kv)  # (q_heads,)
    gate_expanded = gate_expanded.view(1, 1, q_heads, 1)  # broadcastable mask

    # Reshape outputs to (B,S,H,D) for mixing
    full_attn_output_reshaped = full_attn_output.view(bsz, seq_len, q_heads, self.head_dim)
    streaming_attn_output_reshaped = streaming_attn_output.view(bsz, seq_len, q_heads, self.head_dim)

    mixed_streaming = (1 - gate_expanded) * streaming_attn_output_reshaped + gate_expanded * full_attn_output_reshaped

    mixed_streaming = mixed_streaming.view(bsz, seq_len, -1)

    attn_output = torch.cat([full_attn_output, mixed_streaming], dim=0)
    return attn_output, None


# ------------------------- Enable Functions ------------------------- #

def enable_qwen3_duo_attention_training(
    model: Qwen3ForCausalLM,
    sink_size,
    recent_size,
    max_length,
    initial_value=1.0,
    enable_ulysses_attention=False,
    streaming_attn_implementation="blocksparse",
):
    """Attach duo-attention training forward & gating parameters to each attention layer.
    (Streaming impl specifics ignored for simplicity; only head score learning.)
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    for layer in model.model.layers:  # type: ignore
        module = layer.self_attn
        if not hasattr(module, "full_attention_heads"):
             # Qwen3Attention does not have num_key_value_heads, use num_key_value_groups as gating dimension
            module.register_parameter(
                "full_attention_heads",
                nn.Parameter(
                    torch.ones(module.num_key_value_groups, device=device, dtype=dtype) * initial_value
                ),
            )
        module.forward = types.MethodType(qwen3_duo_attention_forward_two_way, module)
        module.sink_size = sink_size  # kept for interface parity
        module.recent_size = recent_size


def enable_qwen3_duo_attention_eval(*args, **kwargs):  # pragma: no cover
    raise NotImplementedError("Evaluation mode not implemented in simplified Qwen3 adapter.")


# ------------------------- Helpers for head params ------------------------- #

def _iter_attention_modules(model):
    if isinstance(model, Qwen3ForCausalLM):
        for layer in model.model.layers:  # type: ignore
            mod = layer.self_attn
            if hasattr(mod, "full_attention_heads"):
                yield mod.full_attention_heads
    elif isinstance(model, Qwen3Model):
        for layer in model.layers:
            mod = layer.self_attn
            if hasattr(mod, "full_attention_heads"):
                yield mod.full_attention_heads
    else:  # FSDP wrapped etc.
        for module in model.modules():
            if hasattr(module, "full_attention_heads"):
                yield module.full_attention_heads


def get_qwen3_full_attention_heads(model):
    return [p for p in _iter_attention_modules(model)]


def set_qwen3_full_attention_heads(model, full_attention_heads):
    idx = 0
    for p in _iter_attention_modules(model):
        if idx >= len(full_attention_heads):
            break
        p.data = full_attention_heads[idx].to(p.device, p.dtype)
        idx += 1
    return model


def map_qwen3_full_attention_heads(model, func):
    for p in _iter_attention_modules(model):
        func(p)
    return model
