import sys
import os
from importlib.metadata import version
# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)
import warnings
import transformers

from adaptive_kv.monkeypatch.adaptive_qwen3_hijack import (
    adaptive_qwen3_Attention_forward,
    adaptive_qwen3_Model_forward, 
    adaptive_quantization_qwen3_Attention_forward
)

from adaptive_kv.monkeypatch.adaptive_llama3_hijack import (
    adaptive_llama_Attention_forward,
    adaptive_llama_Model_forward,
    adaptive_quantization_llama3_Attention_forward
)

from adaptive_kv.monkeypatch.adaptive_mistral_hijack import (
    adaptive_mistral_Attention_forward,
    adaptive_mistral_Model_forward
)

def replace_qwen_adaptive():
    transformers.models.qwen3.modeling_qwen3.Qwen3Attention.forward = adaptive_qwen3_Attention_forward
    transformers.models.qwen3.modeling_qwen3.Qwen3Model.forward = adaptive_qwen3_Model_forward
    
def replace_qwen_quant_adaptive():
    transformers.models.qwen3.modeling_qwen3.Qwen3Attention.forward = adaptive_quantization_qwen3_Attention_forward
    
def replace_llama_adaptive():
    transformers.models.llama.modeling_llama.LlamaAttention.forward = adaptive_llama_Attention_forward
    transformers.models.llama.modeling_llama.LlamaModel.forward = adaptive_llama_Model_forward
    
def replace_llama_quant_adaptive():
    transformers.models.llama.modeling_llama.LlamaAttention.forward = adaptive_llama_Quant_Attention_forward
    
def replace_mistral_adaptive():
    transformers.models.mistral.modeling_mistral.MistralAttention.forward = adaptive_mistral_Attention_forward
    transformers.models.mistral.modeling_mistral.MistralModel.forward = adaptive_mistral_Model_forward

