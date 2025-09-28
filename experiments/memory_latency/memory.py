import sys
import os
from ast import arg
# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)
import site
from datasets import load_dataset
import torch
import json
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
import random
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
import gc
import time
import adaptive_kv
from adaptive_kv.monkeypatch.monkeypatch import replace_qwen_adaptive
from adaptive_kv.monkeypatch.utils import set_cc_size,set_headkv_size,set_snapkv_size
# print('name_space_position:',adaptive_kv.__path__)


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def load_model_and_tokenizer(path):
    tokenizer = AutoTokenizer.from_pretrained(path,
                                              trust_remote_code=True,
                                              )
    model = AutoModelForCausalLM.from_pretrained(path,
                                             torch_dtype=torch.bfloat16,
                                             attn_implementation="flash_attention_2",
                                             device_map="auto",
                                             trust_remote_code=True,
                                             )
    model = model.eval()
    return model, tokenizer

def read_and_concatenate_texts(directory):
    text = ""
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), "r", encoding="utf-8") as f:
                text += f.read() + "\n"
    return text

def truncate_to_max_length(text, tokenizer, max_length):
    tokens = tokenizer.encode(text, truncation=True, max_length=max_length)
    return tokenizer.decode(tokens)

if __name__ == '__main__':
    seed_everything(42)
    MODEL = "/path/to/models/Qwen3-32B"  # Update this path to your model
    haystack_dir = "./adaptive_kv/assets/datasets/PaulGrahamEssays"
    world_size = torch.cuda.device_count()
    mp.set_start_method('spawn', force=True)

    model_name_or_path = MODEL
    model_name = "qwen3"
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['base','fix','ada','pyramid','cokv','headkv','snapkv'], required=True)
    parser.add_argument('--cache_size', type=int, required=True)
    args = parser.parse_args()

    mode = args.mode
    cache_size = args.cache_size

    compress_args = {}
    compress_args_path = f"c{cache_size}_w32_k7_maxpool.json"
    if compress_args_path:
        config_dir = "./experiments/longbench/config"
        compress_args = json.load(open(os.path.join(config_dir, compress_args_path), "r"))
        compress_args['window_size'] = 8
        compress_args['floor_alpha'] = 0.2
        compress_args['gqa_support'] = True
        compress_args['normalize'] = False
        compress_args['pyram_mode']= False
        compress_args['skip'] = 0
        compress_args['pyram_beta'] = 20
        if mode == 'pyramid':
            compress_args['pyram_mode'] = True
        compress_args['mask_heads'] = None
        if mode in ["ada","cokv","headkv","snapkv","pyramid"]:
            replace_qwen_adaptive()
    else:
        print("Base mode")

    def config_compress(model, window_size=32, base_capacity=512, kernel_size=7, pooling="maxpool", floor_alpha=0.5, pyram_mode = False, pyram_beta = 20, normalize=True, skip=0, gqa_support=False,given_adaptive_size=None,mask_heads=None):
        model.model.config.window_size = window_size
        model.model.config.base_capacity = base_capacity
        model.model.config.kernel_size = kernel_size

        model.model.config.normalize = normalize
        model.model.config.pooling = pooling
        model.model.config.floor_alpha = floor_alpha

        model.model.config.pyram_mode = pyram_mode
        model.model.config.pyram_beta = pyram_beta
        model.model.config.skip = skip
        model.model.config.gqa_support = gqa_support
        model.model.config.given_adaptive_size = given_adaptive_size
        model.model.config.mask_heads = mask_heads
        return model

    # NOTE: load model after replace
    model, tokenizer = load_model_and_tokenizer(model_name_or_path)

    text = read_and_concatenate_texts(haystack_dir)

    if compress_args_path:
        model = config_compress(model, **compress_args)
    
    if mode == 'cokv':
        model.model.config.given_adaptive_size = set_cc_size(compress_args['base_capacity'],compress_args['window_size'],model_name, "qasper", 10)
    elif mode == 'headkv':
        model.model.config.given_adaptive_size = set_headkv_size(compress_args['base_capacity'],compress_args['window_size'],model_name,1.01)
    elif mode =='snapkv':
        model.model.config.given_adaptive_size = set_snapkv_size(compress_args['base_capacity'],compress_args['window_size'])

    lengths = [1*1024, 2*1024, 4*1024, 8*1024, 16*1024, 32*1024, 64*1024, 128*1024]
    max_memory_list = []
    for context_length in lengths:
        truncate_text = truncate_to_max_length(text, tokenizer, max_length=context_length)
        input_ids = tokenizer.encode(truncate_text, return_tensors="pt").to(model.device)
        print(f"Testing {mode} {cache_size} length {len(input_ids[0])} ...")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            model.generate(
                input_ids,
                num_beams=1,
                do_sample=False,
                max_new_tokens = 1,  
                temperature=1.0,
                early_stopping=False,      
            )
        device = model.device if hasattr(model, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        max_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        max_memory_list.append(max_memory)
        print(f"Context Length: {context_length}, Memory Used: {max_memory:.2f} MB")
    
    with open(f'{mode}_{cache_size}-1.json', 'w', encoding='utf-8') as file:
        json.dump(max_memory_list, file, indent=4, ensure_ascii=False)