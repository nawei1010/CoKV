import warnings
import copy
import torch
import time
import torch.nn.functional as F
import torch.nn as nn
import math
from typing import List, Optional, Tuple, Union, Any,Dict
from transformers.cache_utils import Cache, DynamicCache, DynamicLayer
from flash_attn import flash_attn_func
import os
import numpy as np
import json
import random
# perform qk calculation and get indices
# this version will not update in inference mode

class DynamicFlattenLayer(DynamicLayer):
    """
    A cache layer that handles flatten head layout, inheriting from DynamicLayer.
    Only overrides the update method to implement flatten logic for per-head caching.
    """
    
    def reset(self) -> None:
        """Resets the cache values while preserving the objects"""
        if self.keys is not None:
            self.keys.zero_()
        if self.values is not None:
            self.values.zero_()
    
    def update(
        self,
        key_states: Union[List[torch.Tensor], torch.Tensor],
        value_states: Union[List[torch.Tensor], torch.Tensor],
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with flatten layer logic for per-head caching.
        
        Parameters:
            key_states: New key states (list of per-head tensors or single tensor)
            value_states: New value states (list of per-head tensors or single tensor)
            cache_kwargs: Additional arguments including "head_lens", "cu_klen"
        
        Returns:
            Tuple of updated key and value states in flattened format
        """
        # Handle first time initialization
        if self.keys is None:
            # First update - initialize with flattened format
            if isinstance(key_states, list):
                # Direct list input - flatten into single tensor
                head_dim = key_states[0].shape[-1] if key_states else 64
                self.keys = torch.cat([k.reshape(-1, head_dim) for k in key_states], dim=0)
                self.values = torch.cat([v.reshape(-1, head_dim) for v in value_states], dim=0)
            else:
                # Check tensor dimensions
                if len(key_states.shape) == 2:
                    # Already flattened format - use directly
                    self.keys = key_states
                    self.values = value_states
                elif len(key_states.shape) == 4:
                    # Standard 4D tensor input - convert to flattened format
                    batch_size, num_heads, seq_len, head_dim = key_states.shape
                    self.keys = key_states.reshape(-1, head_dim)
                    self.values = value_states.reshape(-1, head_dim)
                else:
                    raise ValueError(f"Unexpected key_states shape: {key_states.shape}. Expected 2D or 4D tensor.")
        else:
            # Incremental update with flatten logic
            if cache_kwargs is None:
                raise ValueError("cache_kwargs with 'head_lens' and 'cu_klen' required for incremental updates")
            
            head_lens = cache_kwargs["head_lens"]
            cu_klen = cache_kwargs["cu_klen"]
            
            # Convert input to proper format
            if isinstance(key_states, torch.Tensor):
                batch_size, num_heads, seq_len, head_dim = key_states.shape
                assert batch_size == 1 and seq_len == 1, "Expected batch_size=1 and seq_len=1 for incremental updates"
                
                # Convert to per-head list format for processing
                key_states_list = [key_states[:, i:i+1, :, :] for i in range(num_heads)]
                value_states_list = [value_states[:, i:i+1, :, :] for i in range(num_heads)]
            else:
                key_states_list = key_states
                value_states_list = value_states
                head_dim = key_states_list[0].shape[-1] if key_states_list else 64
            
            # Performance profiling
            import nvtx
            copy_old_rng = nvtx.start_range("flatten_layer_update")
            
            # Use CUDA kernel for efficient flatten update
            from tiny_api_cuda import update_flatten_view
            
            # Flatten new key/value states
            new_keys_flat = torch.cat([k.reshape(-1, head_dim) for k in key_states_list], dim=0)
            new_values_flat = torch.cat([v.reshape(-1, head_dim) for v in value_states_list], dim=0)
            
            # Perform the flatten update operation
            self.keys = update_flatten_view(
                self.keys.reshape(-1, head_dim), 
                new_keys_flat, 
                head_lens, 
                cu_klen
            )
            self.values = update_flatten_view(
                self.values.reshape(-1, head_dim), 
                new_values_flat, 
                head_lens, 
                cu_klen
            )
            
            nvtx.end_range(copy_old_rng)
        
        return self.keys, self.values


class DynamicCacheSplitHeadFlatten(DynamicCache):
    """
    Flattened version of DynamicCache that uses DynamicFlattenLayer for per-head caching.
    """
    def __init__(self, *args, **kwargs) -> None:
        # Initialize parent DynamicCache normally
        super().__init__(*args, **kwargs)
        # Override the layer_classes after initialization to use our custom layer
        self.layer_classes = DynamicFlattenLayer
        
        # Clear any existing layers that were created with the wrong class
        # and recreate them with the correct class
        num_layers_needed = len(self.layers)
        self.layers.clear()
        if num_layers_needed > 0:
            self.append_new_layers(num_layers_needed - 1)

    # All methods are now inherited from DynamicCache base class
    # The flatten logic is handled by DynamicFlattenLayer

# Copied from transformers.models.llama.modeling_llama.repeat_kv for gqa_support
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class H2OKVCluster():
    def __init__(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', layer_idx = None, num_hidden_layers = None, pyram_mode = False, pyram_beta = 20,gqa_support=False,num_key_value_groups = 1):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling

        self.pyram_init = False
        self.pyram_mode = pyram_mode
        self.pyram_beta = pyram_beta
        self.layer_idx = layer_idx
        self.num_hidden_layers = num_hidden_layers
        
        # support gqa
        self.gqa_support = gqa_support
        self.num_key_value_groups = num_key_value_groups
        if self.gqa_support:
            if self.num_key_value_groups == 1:
                warnings.warn("gqa_support is enabled, but num_key_value_groups is 1, which means the model is not using gqa. Please check the model configuration.")

    def reset(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling

    def update_kv(self, origin_key_states, query_states, origin_value_states):
        
        # support gqa
        key_states = repeat_kv(origin_key_states, self.num_key_value_groups)
        value_states = repeat_kv(origin_value_states, self.num_key_value_groups)
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape

        # compute pyramidal capacity
        if self.pyram_mode and not self.pyram_init:
            # NOTE: (max_num + min_num) / 2 == base_capacity to restrict the total capacity
            base_capacity = self.max_capacity_prompt - self.window_size
            min_num = base_capacity // self.pyram_beta
            max_num = base_capacity * 2 - min_num
                
            # if the max_num is larger than the query length, we need to adjust the max_num
            if max_num >= q_len - self.window_size:
                max_num = q_len - self.window_size
                min_num = base_capacity * 2 - max_num
        
            # NOTE: compute interval
            steps = (max_num - min_num) // (self.num_hidden_layers - 1)

            self.max_capacity_prompt = max_num - self.layer_idx * steps + self.window_size
            self.pyram_init = True
            print(f"Pyram mode adaptive capacity, layer: {self.layer_idx}, max_capacity_prompt: {self.max_capacity_prompt}, base_capacity: {self.max_capacity_prompt - self.window_size}")

        if q_len < self.max_capacity_prompt:
            # support gqa
            if self.gqa_support:
                return origin_key_states, origin_value_states
            else:
                return key_states, value_states
        else:
            attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights_mean = attn_weights[:, :, -self.window_size:, : -self.window_size].mean(dim = -2)
            
            # gqa_support 
            if self.gqa_support:
                attn_weights_mean = attn_weights_mean.view(attn_weights_mean.shape[0], -1, self.num_key_value_groups, attn_weights_mean.shape[-1])
                attn_weights_mean = attn_weights_mean.mean(dim=-2)

            indices = attn_weights_mean.topk(self.max_capacity_prompt - self.window_size, dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            
            # support gqa
            if self.gqa_support:
                k_past_compress = origin_key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
                v_past_compress = origin_value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
                k_cur = origin_key_states[:, :, -self.window_size:, :]
                v_cur = origin_value_states[:, :, -self.window_size:, :]
            else:
                k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
                v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
                k_cur = key_states[:, :, -self.window_size:, :]
                v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)
            return key_states, value_states
   

class SnapKVCluster():
    def __init__(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', layer_idx = None, num_hidden_layers = None, pyram_mode = False, pyram_beta = 20,gqa_support=False,num_key_value_groups = 1):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling

        self.pyram_init = False
        self.pyram_mode = pyram_mode
        self.pyram_beta = pyram_beta
        self.layer_idx = layer_idx
        self.num_hidden_layers = num_hidden_layers
        
        # support gqa
        self.gqa_support = gqa_support
        self.num_key_value_groups = num_key_value_groups
        if self.gqa_support:
            if self.num_key_value_groups == 1:
                warnings.warn("gqa_support is enabled, but num_key_value_groups is 1, which means the model is not using gqa. Please check the model configuration.")



    def reset(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling

    def update_kv(self, origin_key_states, query_states, origin_value_states):
        
        # support gqa
        key_states = repeat_kv(origin_key_states, self.num_key_value_groups)
        value_states = repeat_kv(origin_value_states, self.num_key_value_groups)
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape

        # compute pyramidal capacity
        if self.pyram_mode and not self.pyram_init:
            # NOTE: (max_num + min_num) / 2 == base_capacity to restrict the total capacity
            base_capacity = self.max_capacity_prompt - self.window_size
            min_num = base_capacity // self.pyram_beta
            max_num = base_capacity * 2 - min_num
                
            # if the max_num is larger than the query length, we need to adjust the max_num
            if max_num >= q_len - self.window_size:
                max_num = q_len - self.window_size
                min_num = base_capacity * 2 - max_num
        
            # NOTE: compute interval
            steps = (max_num - min_num) // (self.num_hidden_layers - 1)

            self.max_capacity_prompt = max_num - self.layer_idx * steps + self.window_size
            self.pyram_init = True
            print(f"Pyram mode adaptive capacity, layer: {self.layer_idx}, max_capacity_prompt: {self.max_capacity_prompt}, base_capacity: {self.max_capacity_prompt - self.window_size}")

        if q_len < self.max_capacity_prompt:
            # support gqa
            if self.gqa_support:
                return origin_key_states, origin_value_states
            else:
                return key_states, value_states
        else:
            attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights_mean = attn_weights[:, :, -self.window_size:, : -self.window_size].mean(dim = -2)
            
            # gqa_support 
            if self.gqa_support:
                attn_weights_mean = attn_weights_mean.view(attn_weights_mean.shape[0], -1, self.num_key_value_groups, attn_weights_mean.shape[-1])
                attn_weights_mean = attn_weights_mean.mean(dim=-2)

                
            if self.pooling == 'avgpool':
                attn_cache = F.avg_pool1d(attn_weights_mean, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            elif self.pooling == 'maxpool':
                attn_cache = F.max_pool1d(attn_weights_mean, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            else:
                raise ValueError('Pooling method not supported')

            indices = attn_cache.topk(self.max_capacity_prompt - self.window_size, dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            
            # support gqa
            if self.gqa_support:
                k_past_compress = origin_key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
                v_past_compress = origin_value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
                k_cur = origin_key_states[:, :, -self.window_size:, :]
                v_cur = origin_value_states[:, :, -self.window_size:, :]
            else:
                k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
                v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
                k_cur = key_states[:, :, -self.window_size:, :]
                v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)
            return key_states, value_states
   

class AdaptiveSnapKVCluster():
    def __init__(self, window_size = 32, kernel_size = 7, pooling = 'maxpool',base_capacity=None,floor_alpha = None,skip = None,normalize=None, 
                 layer_idx = None, num_hidden_layers = None, pyram_mode = False, pyram_beta = 20,gqa_support=False,num_key_value_groups = 1, given_adaptive_size=None):
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.base_capacity = base_capacity - window_size
        self.floor_ratio = floor_alpha
        self.floor_capacity = int(self.base_capacity * self.floor_ratio)
        self.adaptive_capacity = self.base_capacity - self.floor_capacity
        self.skip = skip

        self.normalize = normalize
        self.pyram_init = False
        self.pyram_mode = pyram_mode
        self.pyram_beta = pyram_beta
        self.layer_idx = layer_idx
        self.num_hidden_layers = num_hidden_layers

        # NOTE: layer-wise meta-data
        self.head_lens = None
        self.max_seqlen_k = 0
        self.klen_sum = 0
        self.cu_klen = 0
        self.cu_qlen = None
        self.cu_offset = None
        self.cu_headlens = None

         # support gqa
        self.gqa_support = gqa_support
        self.num_key_value_groups = num_key_value_groups
        self.given_adaptive_size = given_adaptive_size
        if self.gqa_support:
            if self.num_key_value_groups == 1:
                warnings.warn("gqa_support is enabled, but num_key_value_groups is 1, which means the model is not using gqa. Please check the model configuration.")


    def calcul_attn_sore(self, key_states, query_states):
        bsz, num_heads, q_len, head_dim = query_states.shape
        
        # Add this line: ensure not to exceed actual sequence length
        actual_window_size = min(q_len, self.window_size)

        attn_weights = torch.matmul(query_states[..., -actual_window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
        mask = torch.full((actual_window_size, actual_window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(attn_weights.device)
        attention_mask = mask[None, None, :, :]

        attn_weights[:, :, -actual_window_size:, -actual_window_size:] += attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights_mean = attn_weights[:, :, -actual_window_size:, : -actual_window_size].mean(dim=-2)

        if self.gqa_support:
            attn_weights_mean = attn_weights_mean.view(attn_weights_mean.shape[0],num_heads//self.num_key_value_groups,self.num_key_value_groups,-1)
            attn_weights_mean = attn_weights_mean.mean(dim=-2)

        # if self.pooling == 'avgpool':
        #     attn_weights_mean_pooling = F.avg_pool1d(attn_weights_mean, kernel_size=self.kernel_size,
        #                                              padding=self.kernel_size // 2,
        #                                              stride=1)
        # elif self.pooling == 'maxpool':
        #     attn_weights_mean_pooling = F.max_pool1d(attn_weights_mean, kernel_size=self.kernel_size,
        #                                              padding=self.kernel_size // 2,
        #                                              stride=1)
        else:
            raise ValueError('Pooling method not supported')
        # return attn_weights_mean_pooling
        return attn_weights_mean
    
    def gqa_init_metadata(self,num_heads, k_lens, klen_sum, max_seqlen_k, _device):
        
        # init metadata
        self.head_lens = torch.tensor(k_lens, dtype=torch.int32, device=_device)
        self.klen_sum = klen_sum
        self.max_seqlen_k = max_seqlen_k
        self.cu_headlens = torch.cumsum(self.head_lens, dim=0, dtype=torch.int32)
        # init varlen flash attention metadata
        self.cu_klen = self.cu_headlens - self.head_lens
        self.cu_klen = torch.cat(
            [self.cu_klen, torch.tensor([self.klen_sum], dtype=torch.int32, device=_device)], dim=0)
        # check bug
        self.layer_qlens = torch.ones(num_heads//self.num_key_value_groups, dtype=torch.int32,device=_device)
        self.qlen_sum = num_heads//self.num_key_value_groups
        self.cu_qlen = torch.cumsum(self.layer_qlens, dim=0, dtype=torch.int32) - self.layer_qlens
        self.cu_qlen = torch.cat(
            [self.cu_qlen, torch.tensor([self.qlen_sum], dtype=torch.int32, device=_device)], dim=0)
        if self.gqa_support:
            self.cu_offset = torch.arange(0, num_heads//self.num_key_value_groups + 1, dtype=torch.int32, device=_device)
            self.cu_head_offset = torch.arange(1, num_heads//self.num_key_value_groups +1, dtype=torch.int32, device=_device)

        else:
            self.cu_offset = torch.arange(0, num_heads + 1, dtype=torch.int32, device=_device)
            self.cu_head_offset = torch.arange(1, num_heads+1, dtype=torch.int32, device=_device)
    
    def update_kv(self, origin_key_states, query_states, origin_value_states,layer_idx):
        if self.gqa_support:
            if self.given_adaptive_size == None:
                return self.update_kv_gqa(origin_key_states, query_states, origin_value_states)
            else:
                return self.update_kv_gqa_with_given_adaptive_size(origin_key_states, query_states, origin_value_states, self.given_adaptive_size[layer_idx])
        else:
            return self.update_kv_wo_gqa(origin_key_states, query_states, origin_value_states)


    # update kv with gqa_support
    def update_kv_gqa(self, origin_key_states, query_states, origin_value_states):
        #key_states = origin_key_states
        key_states = repeat_kv(origin_key_states, self.num_key_value_groups)
        #value_states = repeat_kv(origin_value_states, self.num_key_value_groups)

        # check if prefix phase        assert key_states.shape[-2] == query_states.shape[-2]
        _device = key_states.device
        bsz, num_heads, q_len, head_dim = query_states.shape
        attn_score= self.calcul_attn_sore(key_states,query_states)
        origin_heads_key_states = torch.split(origin_key_states, 1, dim=1)
        origin_heads_value_states = torch.split(origin_value_states, 1, dim=1)

        # compute pyramidal capacity
        if self.pyram_mode and not self.pyram_init:
            # NOTE: (max_num + min_num) / 2 == base_capacity to restrict the total capacity
            min_num = self.base_capacity // self.pyram_beta
            max_num = self.base_capacity * 2 - min_num
                
            # if the max_num is larger than the query length, we need to adjust the max_num
            if max_num >= q_len - self.window_size:
                max_num = q_len - self.window_size
                min_num = self.base_capacity * 2 - max_num
        
            # NOTE: compute interval
            steps = (max_num - min_num) // (self.num_hidden_layers - 1)

            # renew adaptive capacity
            self.base_capacity = max_num - self.layer_idx * steps
            self.floor_capacity = int(self.base_capacity * self.floor_ratio)
            self.adaptive_capacity = self.base_capacity - self.floor_capacity
            self.pyram_init = True
            print(f"Pyram mode adaptive capacity, layer: {self.layer_idx}, acap: {self.adaptive_capacity}, bcap: {self.base_capacity}, fcap: {self.floor_capacity}")


        if self.base_capacity > attn_score.size(-1):
            self.gqa_init_metadata(num_heads, [q_len] * (num_heads//self.num_key_value_groups), q_len * (num_heads//self.num_key_value_groups), q_len, _device)
            # not compress
            return origin_key_states.reshape(-1, head_dim), origin_value_states.reshape(-1, head_dim)


        sorted_attn_score,sorted_attn_score_indices = attn_score.sort(dim=-1,descending=True)
        if self.layer_idx >= self.skip:
            adaptive_attn_score = sorted_attn_score
            length = adaptive_attn_score.size(dim=-1)
            if self.normalize:
                ratio_weight = sorted_attn_score[...,:self.base_capacity].sum(dim=-1,keepdim=True)/sorted_attn_score.sum(dim=-1,keepdim=True)
                adaptive_attn_score = adaptive_attn_score*ratio_weight
            adaptive_attn_score = adaptive_attn_score.reshape(bsz,length*num_heads//self.num_key_value_groups)
            sorted_indices = torch.topk(adaptive_attn_score,k=num_heads*self.base_capacity//self.num_key_value_groups,dim=-1).indices
            sorted_indices = sorted_indices//length

            # floor_alpha capacity set
            head_adaptive_capacity = torch.zeros((bsz,num_heads//self.num_key_value_groups),device=_device,dtype = sorted_indices.dtype)
            head_adaptive_capacity.scatter_add_(-1,sorted_indices,torch.ones_like(sorted_indices,dtype=head_adaptive_capacity.dtype),)
            assert head_adaptive_capacity.sum().item() == num_heads*self.base_capacity//self.num_key_value_groups
            head_adaptive_capacity = torch.round(head_adaptive_capacity * (1-self.floor_ratio) + self.floor_capacity).int()
        else:
            head_adaptive_capacity = torch.ones((bsz,num_heads),device=_device,dtype = sorted_attn_score_indices.dtype) * self.base_capacity

        sorted_attn_score_indices = sorted_attn_score_indices.split(1,dim=1)

        heads_key_states = []
        heads_value_states = []
        assert bsz == 1
        # per head

        # reinit varlen metadata
        k_lens = []
        klen_sum = 0
        max_seqlen_k = 0
        self.cu_klen = 0

        #print(self.layer_idx,head_adaptive_capacity)


        for head_idx in range(num_heads//self.num_key_value_groups):
            cache_index = sorted_attn_score_indices[head_idx][...,:head_adaptive_capacity[0][head_idx]]

            l = cache_index.shape[-1] + self.window_size
            k_lens.append(l)
            max_seqlen_k = max(max_seqlen_k, l)
            klen_sum += l

            cache_index = cache_index.view(1, 1, -1, 1).expand(-1, -1, -1, head_dim)
            top_Kcache = origin_heads_key_states[head_idx].gather(dim=2,index=cache_index)
            top_Vcache = origin_heads_value_states[head_idx].gather(dim=2,index=cache_index)
            selected_k = torch.cat([top_Kcache,origin_heads_key_states[head_idx][:, :, -self.window_size:, :]],dim=2)
            selected_v = torch.cat([top_Vcache,origin_heads_value_states[head_idx][:, :, -self.window_size:, :]],dim=2)

            # NOTE: flatten view
            heads_key_states.append(selected_k.reshape(-1, head_dim))
            heads_value_states.append(selected_v.reshape(-1, head_dim))

        self.gqa_init_metadata(num_heads, k_lens, klen_sum, max_seqlen_k, _device)

        # NOTE: compose as flatten view
        heads_key_states = torch.cat(heads_key_states, dim=0)
        heads_value_states = torch.cat(heads_value_states, dim=0)

        return heads_key_states, heads_value_states

    
    # update without gqa_support
    def update_kv_wo_gqa(self,  origin_key_states, query_states, origin_value_states):
        key_states = repeat_kv(origin_key_states, self.num_key_value_groups)
        value_states = repeat_kv(origin_value_states, self.num_key_value_groups)

        # check if prefix phase        assert key_states.shape[-2] == query_states.shape[-2]
        _device = key_states.device
        bsz, num_heads, q_len, head_dim = query_states.shape
        attn_score= self.calcul_attn_sore(key_states,query_states)
        origin_heads_key_states = torch.split(key_states, 1, dim=1)
        origin_heads_value_states = torch.split(value_states, 1, dim=1)

        # compute pyramidal capacity
        if self.pyram_mode and not self.pyram_init:
            # NOTE: (max_num + min_num) / 2 == base_capacity to restrict the total capacity
            min_num = self.base_capacity // self.pyram_beta
            max_num = self.base_capacity * 2 - min_num
                
            # if the max_num is larger than the query length, we need to adjust the max_num
            if max_num >= q_len - self.window_size:
                max_num = q_len - self.window_size
                min_num = self.base_capacity * 2 - max_num
        
            # NOTE: compute interval
            steps = (max_num - min_num) // (self.num_hidden_layers - 1)

            # renew adaptive capacity
            self.base_capacity = max_num - self.layer_idx * steps
            self.floor_capacity = int(self.base_capacity * self.floor_ratio)
            self.adaptive_capacity = self.base_capacity - self.floor_capacity
            self.pyram_init = True
            print(f"Pyram mode adaptive capacity, layer: {self.layer_idx}, acap: {self.adaptive_capacity}, bcap: {self.base_capacity}, fcap: {self.floor_capacity}")

        def init_metadata(num_heads, k_lens, klen_sum, max_seqlen_k):
            # init metadata
            self.head_lens = torch.tensor(k_lens, dtype=torch.int32, device=_device)
            self.klen_sum = klen_sum
            self.max_seqlen_k = max_seqlen_k
            self.cu_headlens = torch.cumsum(self.head_lens, dim=0, dtype=torch.int32)
            # init varlen flash attention metadata
            self.cu_klen = self.cu_headlens - self.head_lens
            self.cu_klen = torch.cat(
                [self.cu_klen, torch.tensor([self.klen_sum], dtype=torch.int32, device=_device)], dim=0)
            self.layer_qlens = torch.ones(num_heads, dtype=torch.int32,device=_device)
            self.qlen_sum = num_heads
            self.cu_qlen = torch.cumsum(self.layer_qlens, dim=0, dtype=torch.int32) - self.layer_qlens
            self.cu_qlen = torch.cat(
                [self.cu_qlen, torch.tensor([self.qlen_sum], dtype=torch.int32, device=_device)], dim=0)
            self.cu_offset = torch.arange(0, num_heads + 1, dtype=torch.int32, device=_device)
            self.cu_head_offset = torch.arange(1, num_heads+1, dtype=torch.int32, device=_device)

        if self.base_capacity > attn_score.size(-1):
            init_metadata(num_heads, [q_len] * num_heads, q_len * num_heads, q_len)
            # not compress
            return key_states.reshape(-1, head_dim), value_states.reshape(-1, head_dim)

        # if you need to weight the attn_score
        pass
        sorted_attn_score,sorted_attn_score_indices = attn_score.sort(dim=-1,descending=True)
        if self.layer_idx >= self.skip:
            adaptive_attn_score = sorted_attn_score
            length = adaptive_attn_score.size(dim=-1)
            if self.normalize:
                ratio_weight = sorted_attn_score[...,:self.base_capacity].sum(dim=-1,keepdim=True)/sorted_attn_score.sum(dim=-1,keepdim=True)
                adaptive_attn_score = adaptive_attn_score*ratio_weight
            adaptive_attn_score = adaptive_attn_score.reshape(bsz,length*num_heads)
            sorted_indices = torch.topk(adaptive_attn_score,k=num_heads*self.base_capacity,dim=-1).indices
            sorted_indices = sorted_indices//length
            # floor_alpha capacity set
            head_adaptive_capacity = torch.zeros((bsz,num_heads),device=_device,dtype = sorted_indices.dtype)
            head_adaptive_capacity.scatter_add_(-1,sorted_indices,torch.ones_like(sorted_indices,dtype=head_adaptive_capacity.dtype),)
            assert head_adaptive_capacity.sum().item() == num_heads*self.base_capacity
            head_adaptive_capacity = torch.round(head_adaptive_capacity * (1-self.floor_ratio) + self.floor_capacity).int()
        else:
            head_adaptive_capacity = torch.ones((bsz,num_heads),device=_device,dtype = sorted_attn_score_indices.dtype) * self.base_capacity
        sorted_attn_score_indices = sorted_attn_score_indices.split(1,dim=1)

        heads_key_states = []
        heads_value_states = []
        assert bsz == 1
        # per head

        # reinit varlen metadata
        k_lens = []
        klen_sum = 0
        max_seqlen_k = 0
        self.cu_klen = 0


        for head_idx in range(num_heads):
            cache_index = sorted_attn_score_indices[head_idx][...,:head_adaptive_capacity[0][head_idx]]

            l = cache_index.shape[-1] + self.window_size
            k_lens.append(l)
            max_seqlen_k = max(max_seqlen_k, l)
            klen_sum += l

            cache_index = cache_index.view(1, 1, -1, 1).expand(-1, -1, -1, head_dim)
            top_Kcache = origin_heads_key_states[head_idx].gather(dim=2,index=cache_index)
            top_Vcache = origin_heads_value_states[head_idx].gather(dim=2,index=cache_index)
            selected_k = torch.cat([top_Kcache,origin_heads_key_states[head_idx][:, :, -self.window_size:, :]],dim=2)
            selected_v = torch.cat([top_Vcache,origin_heads_value_states[head_idx][:, :, -self.window_size:, :]],dim=2)

            # NOTE: flatten view
            heads_key_states.append(selected_k.reshape(-1, head_dim))
            heads_value_states.append(selected_v.reshape(-1, head_dim))

        init_metadata(num_heads, k_lens, klen_sum, max_seqlen_k)

        # NOTE: compose as flatten view
        heads_key_states = torch.cat(heads_key_states, dim=0)
        heads_value_states = torch.cat(heads_value_states, dim=0)

        return heads_key_states,heads_value_states
    
    # update kv with gqa_support
    def update_kv_gqa_with_given_adaptive_size(self, origin_key_states, query_states, origin_value_states, head_cache_size=None):

        #key_states = origin_key_states
        key_states = repeat_kv(origin_key_states, self.num_key_value_groups)
        #value_states = repeat_kv(origin_value_states, self.num_key_value_groups)

        # check if prefix phase        assert key_states.shape[-2] == query_states.shape[-2]
        _device = key_states.device
        bsz, num_heads, q_len, head_dim = query_states.shape
        attn_score= self.calcul_attn_sore(key_states,query_states)
        origin_heads_key_states = torch.split(origin_key_states, 1, dim=1)
        origin_heads_value_states = torch.split(origin_value_states, 1, dim=1)

        sorted_attn_score,sorted_attn_score_indices = attn_score.sort(dim=-1,descending=True)
        sorted_attn_score_indices = sorted_attn_score_indices.split(1,dim=1)


        if self.base_capacity > attn_score.size(-1):
            self.gqa_init_metadata(num_heads, [q_len] * (num_heads//self.num_key_value_groups), q_len * (num_heads//self.num_key_value_groups), q_len, _device)
            # not compress
            return origin_key_states.reshape(-1, head_dim), origin_value_states.reshape(-1, head_dim)

        heads_key_states = []
        heads_value_states = []
        assert bsz == 1
        # per head

        # reinit varlen metadata
        k_lens = []
        klen_sum = 0
        max_seqlen_k = 0
        self.cu_klen = 0

  
        for head_idx in range(num_heads//self.num_key_value_groups):
            cache_index = sorted_attn_score_indices[head_idx][..., :head_cache_size[head_idx].int()]
            l = cache_index.shape[-1] + self.window_size
            k_lens.append(l)

            max_seqlen_k = max(max_seqlen_k, l)
            klen_sum += l

            cache_index = cache_index.view(1, 1, -1, 1).expand(-1, -1, -1, head_dim)
            top_Kcache = origin_heads_key_states[head_idx].gather(dim=2,index=cache_index)
            top_Vcache = origin_heads_value_states[head_idx].gather(dim=2,index=cache_index)
            selected_k = torch.cat([top_Kcache,origin_heads_key_states[head_idx][:, :, -self.window_size:, :]],dim=2)
            selected_v = torch.cat([top_Vcache,origin_heads_value_states[head_idx][:, :, -self.window_size:, :]],dim=2)

            # NOTE: flatten view
            heads_key_states.append(selected_k.reshape(-1, head_dim))
            heads_value_states.append(selected_v.reshape(-1, head_dim))

        self.gqa_init_metadata(num_heads, k_lens, klen_sum, max_seqlen_k, _device)

        # NOTE: compose as flatten view
        heads_key_states = torch.cat(heads_key_states, dim=0)
        heads_value_states = torch.cat(heads_value_states, dim=0)

        return heads_key_states, heads_value_states


def init_h2okv(self):

    assert hasattr(self.config, 'window_size'), "window_size not set"
    assert hasattr(self.config, 'kernel_size'), "kernel_size not set"
    assert hasattr(self.config, "pooling"), "pooling not set"
    assert hasattr(self.config, "base_capacity"), "base_capacity not set"
    # init only once
    if not hasattr(self, "kv_cluster"):
        self.kv_cluster = H2OKVCluster(
            window_size = self.config.window_size, 
            max_capacity_prompt = self.config.base_capacity,
            layer_idx = self.layer_idx,
            num_hidden_layers = self.config.num_hidden_layers,
            pyram_mode = self.config.pyram_mode,
            pyram_beta = getattr(self.config, 'pyram_beta', 20),
            gqa_support = self.config.gqa_support,
            num_key_value_groups = self.config.num_attention_heads // self.config.num_key_value_heads,
            )
        if self.config.gqa_support:
            if self.config.model_type != "mistral":
                warnings.warn("GQA currently supports only for mistral-7B-v0.2 model")


def init_snapkv(self):

    assert hasattr(self.config, 'window_size'), "window_size not set"
    assert hasattr(self.config, 'kernel_size'), "kernel_size not set"
    assert hasattr(self.config, "pooling"), "pooling not set"
    assert hasattr(self.config, "base_capacity"), "base_capacity not set"
    # init only once
    if not hasattr(self, "kv_cluster"):
        self.kv_cluster = SnapKVCluster(
            window_size = self.config.window_size, 
            max_capacity_prompt = self.config.base_capacity,
            kernel_size = self.config.kernel_size,
            pooling = self.config.pooling,
            layer_idx = self.layer_idx,
            num_hidden_layers = self.config.num_hidden_layers,
            pyram_mode = self.config.pyram_mode,
            pyram_beta = getattr(self.config, 'pyram_beta', 20),
            gqa_support = self.config.gqa_support,
            num_key_value_groups = self.config.num_attention_heads // self.config.num_key_value_heads,
            )
        if self.config.gqa_support:
            if self.config.model_type != "mistral":
                warnings.warn("GQA currently supports only for mistral-7B-v0.2 model")
        # if len(self.config.skip) > 0:
        #     warnings.warn("vanilla transformer should not enable skip",self.config.skip)
        # print(f"Compress config(Snap): window_size={self.kv_cluster.window_size}, max_capacity_prompt={self.kv_cluster.max_capacity_prompt}, kernel_size={self.kv_cluster.kernel_size}, pooling={self.kv_cluster.pooling}, pyram_mode={self.kv_cluster.pyram_mode}, beta={self.kv_cluster.pyram_beta}")

def init_adaptive_snapkv(self):
    assert hasattr(self.config,'window_size'),"window_size not set"
    assert hasattr(self.config,'kernel_size'),"kernel_size not set"
    assert hasattr(self.config,"pooling"),"pooling not set"
    assert hasattr(self.config, "base_capacity"), "base_capacity not set"
    assert hasattr(self.config,"floor_alpha"),"floor_alpha not set"
    assert self.config.floor_alpha is not None


    # Normalize given_adaptive_size to a torch int32 tensor with expected shape
    if hasattr(self.config, "given_adaptive_size") and self.config.given_adaptive_size is not None:
        if not isinstance(self.config.given_adaptive_size, torch.Tensor):
            self.config.given_adaptive_size = torch.as_tensor(self.config.given_adaptive_size, dtype=torch.int32)
        else:
            self.config.given_adaptive_size = self.config.given_adaptive_size.to(dtype=torch.int32)
        expected_shape = (
            self.config.num_hidden_layers,
            self.config.num_attention_heads // self.config.num_key_value_heads,
        )
        assert tuple(self.config.given_adaptive_size.shape) == expected_shape, (
            f"given_adaptive_size shape {tuple(self.config.given_adaptive_size.shape)} does not match "
            f"expected {expected_shape} (num_layers, num_kv_heads)."
        )

    # init only once
    if not hasattr(self, "kv_cluster"):
        self.kv_cluster = AdaptiveSnapKVCluster(
            window_size = self.config.window_size,
            base_capacity=self.config.base_capacity,
            kernel_size = self.config.kernel_size,
            pooling = self.config.pooling,
            floor_alpha= self.config.floor_alpha,
            skip = getattr(self.config, 'skip', None),
            layer_idx = self.layer_idx,
            normalize = self.config.normalize,
            num_hidden_layers = self.config.num_hidden_layers,
            pyram_mode = self.config.pyram_mode,
            pyram_beta = getattr(self.config, 'pyram_beta', 20),
            gqa_support = self.config.gqa_support,
            num_key_value_groups = self.config.num_attention_heads // self.config.num_key_value_heads,
            given_adaptive_size = self.config.given_adaptive_size
            )

    #the adaptive size will change when calculate Shapley value
    elif self.config.given_adaptive_size != None and not torch.all(self.kv_cluster.given_adaptive_size == self.config.given_adaptive_size):
        self.kv_cluster.given_adaptive_size = self.config.given_adaptive_size
        
        
        
def set_headkv_size(base_capability, window_size, model, beta):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    path = os.path.join(project_root, 'adaptive_kv/assets/head_scores/headkv_scores/Qwen3-32B_retrieval_reasoning_heads.json')

    with open(path, 'r') as file:
        head_list = json.loads(file.readline())
        head_score_list = [np.mean(l[1]) for l in head_list.items()]
        head_score_list = torch.tensor(head_score_list / sum(head_score_list))

        temp = 1
        base_capacity=base_capability-window_size
        num_hidden_layers = 64
        num_attention_heads = 64
        beta = beta

        head_score_list = torch.pow(head_score_list, temp)
    
        head_score_list = head_score_list / torch.sum(head_score_list)
        total_attention = head_score_list.reshape(num_hidden_layers, num_attention_heads)
        total_pool_capacity = (base_capacity // beta) * num_hidden_layers * num_attention_heads
        min_num = (base_capacity - base_capacity // beta)
        head_capacity = total_attention * total_pool_capacity + min_num

        cache_size = torch.zeros(64,8)
        for i in range(64):
            for j in range(8):
                for k in range(8):
                    cache_size[i][j] += head_capacity[i][j*8+k]
                cache_size[i][j] = cache_size[i][j]/8

    return np.floor(cache_size)


def set_cc_size(base_capability, window_size, model, dataset, k_th):
    base_capability = base_capability-window_size
    sum_capability = base_capability*64*8
    cache_size = torch.zeros(64,8,dtype=torch.int32)

    cc_left = torch.zeros(513)
    count_left = torch.zeros(513)
    Avg_cc = torch.zeros(513)

    sizes = [256]
    
    for size in sizes:
        data = np.load(f'.//adaptive_kv/assets/head_scores/complementary_contributions/cc_{model}_{dataset}_size_{size}.npy', allow_pickle=True).item()
        # Get the saved arrays
        cc_left = data['cc_left']
        count_left = data['count_left']
        Avg_cc += cc_left/count_left
    Avg_cc = Avg_cc[:512]
    CC = Avg_cc.view(64,8)

    def search_kth_smallest(arr, k):
        flat_list = [element for row in arr for element in row]
        flat_list.sort()
        # Return the k-th smallest element
        kth_smallest = copy.deepcopy(flat_list[k - 1])
        return kth_smallest
    
    border = search_kth_smallest(CC, k_th)

    for i in range(64):
        for j in range(8):
            CC[i][j] = max(CC[i][j]-border,0)
    sum_div = 0
    for layer in range(64):
        for group in range(8):
            sum_div = sum_div+CC[layer][group]
    for layer in range(64):
        for group in range(8):
            cache_size[layer][group] += int(CC[layer][group]/sum_div*sum_capability)

    return np.floor(cache_size)



def set_snapkv_size(base_capability, window_size):
    base_capability = base_capability-window_size
    cache_size = torch.ones(64,8,dtype=torch.int32)*base_capability
    return np.floor(cache_size)

def cc_mask_average(model, flag, k_th,size=20000):
    cache_size = torch.ones(64,8,dtype=torch.int32)*size

    cc_left = torch.zeros(513)
    count_left = torch.zeros(513)
    Avg_cc = torch.zeros(513)

    sizes = [256]
    
    for size in sizes:
        data = np.load(f'.//adaptive_kv/assets/head_scores/complementary_contributions/cc_{model}_average_size_{size}.npy', allow_pickle=True).item()
        # Get the saved arrays
        cc_left = data['cc_left']
        count_left = data['count_left']
        Avg_cc += cc_left/count_left
    Avg_cc = Avg_cc[:512]
    CC = Avg_cc.view(64,8)

    def search_kth_smallest(arr, k):
        flat_list = [element for row in arr for element in row]
        flat_list.sort()
        # Return the k-th smallest element
        kth_smallest = copy.deepcopy(flat_list[k - 1])
        return kth_smallest
    
    if flag == 'large':
        border = search_kth_smallest(CC, 512-k_th)
        for i in range(64):
            for j in range(8):
                if CC[i][j] >= border:
                    cache_size[i][j] = 0
    elif flag == 'small':
        border = search_kth_smallest(CC, k_th)
        for i in range(64):
            for j in range(8):
                if CC[i][j] <= border:
                    cache_size[i][j] = 0
    return np.floor(cache_size)

def cc_mask(model, dataset, flag, k_th,size=20000):
    cache_size = torch.ones(64,8,dtype=torch.int32)*size

    cc_left = torch.zeros(513)
    count_left = torch.zeros(513)
    Avg_cc = torch.zeros(513)

    sizes = [256]
    
    for size in sizes:
        data = np.load(f'.//adaptive_kv/assets/head_scores/complementary_contributions/cc_{model}_{dataset}_size_{size}.npy', allow_pickle=True).item()
        # Get the saved arrays
        cc_left = data['cc_left']
        count_left = data['count_left']
        Avg_cc += cc_left/count_left
    Avg_cc = Avg_cc[:512]
    CC = Avg_cc.view(64,8)

    def search_kth_smallest(arr, k):
        flat_list = [element for row in arr for element in row]
        flat_list.sort()
        # Return the k-th smallest element
        kth_smallest = copy.deepcopy(flat_list[k - 1])
        return kth_smallest
    
    if flag == 'large':
        border = search_kth_smallest(CC, 512-k_th)
        for i in range(64):
            for j in range(8):
                if CC[i][j] >= border:
                    cache_size[i][j] = 0
    elif flag == 'small':
        border = search_kth_smallest(CC, k_th)
        for i in range(64):
            for j in range(8):
                if CC[i][j] <= border:
                    cache_size[i][j] = 0
    print(k_th)
    print(cache_size)
    return np.floor(cache_size)

def headkv_mask(model, flag, k_th, size = 20000):
    cache_size = torch.ones(64,8,dtype=torch.int32)*size
    path = './/adaptive_kv/assets/head_scores/headkv_scores/Qwen3-32B_retrieval_reasoning_heads.json'

    with open(path, 'r') as file:
        head_list = json.loads(file.readline())
        head_score_list = [np.mean(l[1]) for l in head_list.items()]
        head_score_list = torch.tensor(head_score_list / sum(head_score_list))
        num_hidden_layers = 64
        num_attention_heads = 64
        total_attention = head_score_list.reshape(num_hidden_layers, num_attention_heads)
        group_score = torch.zeros(64,8)
        for i in range(64):
            for j in range(8):
                for k in range(8):
                    group_score[i][j] += total_attention[i][j*8+k]

        def search_kth_smallest(arr, k):
            flat_list = [element for row in arr for element in row]
            flat_list.sort()
            # Return the k-th smallest element
            kth_smallest = copy.deepcopy(flat_list[k - 1])
            return kth_smallest
        
        if flag == 'large':
            border = search_kth_smallest(group_score, 512-k_th)
            for i in range(64):
                for j in range(8):
                    if group_score[i][j] >= border:
                        cache_size[i][j] = 0
        elif flag == 'small':
            border = search_kth_smallest(group_score, k_th)
            for i in range(64):
                for j in range(8):
                    if group_score[i][j] <= border:
                        cache_size[i][j] = 0
    print(k_th)
    print(cache_size)
    return np.floor(cache_size)

def random_mask(del_group,size=20000):
    cache_size = torch.ones(64,8,dtype=torch.int32)*size
    numbers = random.sample(range(512), del_group)
    for num in numbers:
        cache_size[int(num/8)][num%8] = 0
    print(cache_size)
    return np.floor(cache_size)

def set_random_bits(high_bits, low_bits):
    cache_size = torch.ones(64,8,dtype=torch.int32)*high_bits
    numbers = random.sample(range(512), 256)
    for num in numbers:
        cache_size[int(num/8)][num%8] = low_bits
    return np.floor(cache_size)

def set_headkv_bits(high_bits,low_bits):
    base_capability = 128
    window_size = 8
    beta = 1.01
    path = './/adaptive_kv/assets/head_scores/headkv_scores/Qwen3-32B_retrieval_reasoning_heads.json'

    with open(path, 'r') as file:
        head_list = json.loads(file.readline())
        head_score_list = [np.mean(l[1]) for l in head_list.items()]
        head_score_list = torch.tensor(head_score_list / sum(head_score_list))

        temp = 1
        base_capacity=base_capability-window_size
        num_hidden_layers = 64
        num_attention_heads = 64
        beta = beta

        head_score_list = torch.pow(head_score_list, temp)
    
        head_score_list = head_score_list / torch.sum(head_score_list)
        total_attention = head_score_list.reshape(num_hidden_layers, num_attention_heads)
        total_pool_capacity = (base_capacity // beta) * num_hidden_layers * num_attention_heads
        min_num = (base_capacity - base_capacity // beta)
        head_capacity = total_attention * total_pool_capacity + min_num

        cache_size = torch.zeros(64,8)
        for i in range(64):
            for j in range(8):
                for k in range(8):
                    cache_size[i][j] += head_capacity[i][j*8+k]
                cache_size[i][j] = cache_size[i][j]/8
                
        average_importance = torch.median(cache_size)  
        kv_bits = torch.zeros(64,8,dtype=torch.int32)      
        for layer in range(64):
            for group in range(8):
                if cache_size[layer][group] >= average_importance:
                    kv_bits[layer][group] = high_bits
                else:
                    kv_bits[layer][group] = low_bits

    return np.floor(kv_bits)

def set_sv_bits(model, dataset, high_bits, low_bits):
    cc = torch.zeros(513)
    count = torch.zeros(513)
    Avg_cc = torch.zeros(513)

    sizes = [256]
    
    for size in sizes:
        data = np.load(f'.//adaptive_kv/assets/head_scores/complementary_contributions/cc_{model}_{dataset}_size_{size}.npy', allow_pickle=True).item()
        # Get the saved arrays
        cc += data['cc_left']
        count += data['count_left']
        cc += data['cc_right']
        count += data['count_right']
        Avg_cc += cc/count
    Avg_cc = Avg_cc[:512]
    CC = Avg_cc.view(64,8)
    
    average_importance = torch.median(CC)
    
    cache_size = torch.zeros(64,8,dtype=torch.int32)
    
    for layer in range(64):
        for group in range(8):
            if CC[layer][group] >= average_importance:
                cache_size[layer][group] = high_bits
            else:
                cache_size[layer][group] = low_bits
                
    return cache_size
