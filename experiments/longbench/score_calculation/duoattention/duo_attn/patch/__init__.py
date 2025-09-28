from .qwen3 import (
    enable_qwen3_duo_attention_training,
    enable_qwen3_duo_attention_eval,
    get_qwen3_full_attention_heads,
    set_qwen3_full_attention_heads,
    map_qwen3_full_attention_heads,
)

import numpy as np
import os
import torch


def enable_duo_attention_training(
    model,
    sink_size,
    recent_size,
    max_length,
    initial_value=1.0,
    enable_ulysses_attention=False,
    streaming_attn_implementation="blocksparse",
):
    print(
        f"Enabling DuoAttention training using {streaming_attn_implementation} implementation"
    )
    if "qwen3" in model.config.model_type.lower():
        enable_qwen3_duo_attention_training(
            model,
            sink_size,
            recent_size,
            max_length,
            initial_value=initial_value,
            enable_ulysses_attention=enable_ulysses_attention,
            streaming_attn_implementation=streaming_attn_implementation,
        )
    else:
        raise ValueError(f"Model type {model.config.model_type} not supported. Only Qwen3 is supported in this simplified version.")


def enable_duo_attention_eval(
    model,
    full_attention_heads,
    sink_size,
    recent_size,
):
    print(
        f"Enabling DuoAttention evaluation using sink size {sink_size} and recent size {recent_size}"
    )
    if "qwen3" in model.config.model_type.lower():
        enable_qwen3_duo_attention_eval(
            model,
            full_attention_heads,
            sink_size,
            recent_size,
        )
    else:
        raise ValueError(f"Model type {model.config.model_type} not supported. Only Qwen3 is supported in this simplified version.")


def get_model_config(model):
    """Helper function to get model config from FSDP wrapped or regular model."""
    if hasattr(model, 'config'):
        return model.config
    elif hasattr(model, '_fsdp_wrapped_module') and hasattr(model._fsdp_wrapped_module, 'config'):
        return model._fsdp_wrapped_module.config
    elif hasattr(model, 'module') and hasattr(model.module, 'config'):
        return model.module.config
    else:
        raise AttributeError("Cannot find model config")


def get_full_attention_heads(model):
    config = get_model_config(model)
    if "qwen3" in config.model_type.lower():
        return get_qwen3_full_attention_heads(model)
    else:
        raise ValueError(f"Model type {config.model_type} not supported. Only Qwen3 is supported in this simplified version.")


def set_full_attention_heads(model, full_attention_heads):
    config = get_model_config(model)
    if "qwen3" in config.model_type.lower():
        model = set_qwen3_full_attention_heads(model, full_attention_heads)
    else:
        raise ValueError(f"Model type {config.model_type} not supported. Only Qwen3 is supported in this simplified version.")
    return model


def map_full_attention_heads(model, func):
    config = get_model_config(model)
    if "qwen3" in config.model_type.lower():
        return map_qwen3_full_attention_heads(model, func)
    else:
        raise ValueError(f"Model type {config.model_type} not supported. Only Qwen3 is supported in this simplified version.")


def load_full_attention_heads(load_dir, filename="full_attention_heads.tsv"):
    full_attention_heads = np.loadtxt(
        os.path.join(load_dir, filename),
        dtype=float,
        delimiter="\t",
    )
    full_attention_heads = np.clip(full_attention_heads, 0, 1)
    full_attention_heads = torch.tensor(full_attention_heads, dtype=torch.float32)
    return full_attention_heads
