import sys
project_path = 
sys.path.append(project_path)
from ast import arg
import sys
import os
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
from tqdm import trange
from adaptive_kv.monkeypatch.monkeypatch import replace_llama_adaptive
from adaptive_kv.monkeypatch.monkeypatch import replace_mistral_adaptive

from metrics import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
)

dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", '--model_name_or_path', type=str, default="path_to_your_model/Llama-3-8B-Instruct")
    parser.add_argument('--max_length', type=int, default=8000)
    parser.add_argument("-d", '--dataset', type=str, default="narrativeqa")
    parser.add_argument('--set_size', type=int, default=0, help='a number')
    parser.add_argument('--sampling_number', type=int, default=1, help='a number') 
    parser.add_argument('--compress_args_path', type=str, default=None, help="Path to the compress args")
    parser.add_argument('--gpu_idx', type=int, default=1, help='a number') 
    parser.add_argument("--skip",type=int, default=0, help="skip layer number")
    parser.add_argument('--mode', type=str, choices=['ada', 'fix', 'base','cokv','sv'], help="Ada mode, fix mode or normal")
    parser.add_argument('--model',type=str,default='llama')
    parser.add_argument('--gqa_support',action='store_true', default=False, help="init gqa_support")
    parser.add_argument('--floor_alpha',type=float,default=0.2,help="floor_alpha budgets for each head")
    parser.add_argument('--normalize',action='store_true')
    parser.add_argument('--pyram',action='store_true',help="using pyram mode")
    parser.add_argument('--pyram_beta',default=20,type=int, help="hyper parameter for pyram")
    return parser.parse_args(args)

# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    if "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "llama2" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    return prompt

def get_pred(model, tokenizer, data, max_length, max_gen, prompt_format, dataset, device, model_name_or_path):
    device = "cuda"
    json_data_list = []
    total_score = 0
    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        # if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
        #     prompt = build_chat(tokenizer, prompt, model_name_or_path)
        
        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input.input_ids.shape[-1]
        if dataset == "samsum": # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length+1,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
            )[0]
        else:
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                eos_token_id=[tokenizer.eos_token_id],
            )[0]
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        json_data = {"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}
        json_data_list.append(json_data)
        gc.collect()
        torch.cuda.empty_cache()

        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            pred = pred.lstrip('\n').split('\n')[0]
        for ground_truth in json_obj["answers"]:
            if dataset == 'lcc1':
                dataset = 'lcc'
            score = max(score, dataset2metric[dataset](pred, ground_truth, all_classes=json_obj["all_classes"]))
        total_score += score
    return total_score/(len(data))
    


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def load_maskmodel_and_tokenizer(path):
    tokenizer = AutoTokenizer.from_pretrained(path,
                                              trust_remote_code=True,
                                              )
    model = AutoModelForCausalLM.from_pretrained(path,
                                            #  torch_dtype=torch.bfloat16,
                                             torch_dtype=torch.bfloat16,
                                             # TODO: hard code
                                             device_map="auto",
                                             attn_implementation="flash_attention_2",
                                             trust_remote_code=True,
                                             )
    model = model.eval()
    return model, tokenizer

if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()
    world_size = torch.cuda.device_count()
    mp.set_start_method('spawn', force=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name_or_path = args.model_name_or_path
    model_name = args.model_name_or_path.split("/")[-1]
    # define your model
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open(f"{project_path}/experiments/LongBench/config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open(f"{project_path}/experiments/LongBench/config/dataset2maxlen.json", "r"))


    # NOTE: Compress config
    compress_args = {}
    compress_args = json.load(open(os.path.join('config', args.compress_args_path), "r"))
    compress_args['window_size'] = 8
    compress_args['floor_alpha'] = args.floor_alpha
    compress_args['gqa_support'] = args.gqa_support
    compress_args['normalize'] = args.normalize
    compress_args['pyram_mode']= args.pyram
    compress_args['skip'] = args.skip
    compress_args['pyram_beta'] = args.pyram_beta
    compress_args['given_adaptive_size'] = torch.ones((32,8))*120
          

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
        return model

    if args.model == 'llama':
        replace_llama_adaptive()
    elif args.model == 'mistral':
        replace_mistral_adaptive()

    # NOTE: load model after replace
    model, tokenizer = load_maskmodel_and_tokenizer(model_name_or_path)
    model = config_compress(model, **compress_args)

    
    data = load_dataset('json', data_files={'test': f"{project_path}/assets/datasets/LongBench-valid/{args.dataset}.jsonl"}, split='test') 
    if args.dataset == 'lcc1':
        prompt_format = dataset2prompt['lcc']
        max_gen = dataset2maxlen['lcc']
    else:
        prompt_format = dataset2prompt[args.dataset]
        max_gen = dataset2maxlen[args.dataset]
    data_all = [data_sample for data_sample in data]
    max_length = args.max_length


    output_dir = f"{project_path}/experiments/LongBench/importance_evaluation/complementary_contributions"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, "cc_{}_{}_{}_gpu{}.npy".format(args.model, args.dataset, args.set_size, args.gpu_idx))
    

    n = 256 
    local_state = np.random.RandomState(None)
    cc_left = np.zeros(n+1)
    cc_right = np.zeros(n+1)
    count_left = np.zeros(n+1)
    count_right = np.zeros(n+1)
    idxs = np.arange(n)
    for _ in trange(args.sampling_number):
        local_state.shuffle(idxs)
        start_idx = args.set_size
        masked_size = torch.zeros((32,8))
        for i in range(start_idx):
            masked_size[int(idxs[i]/8)][idxs[i]%8] = 8000
        model.model.config.given_adaptive_size = masked_size
        u_left = get_pred(model, tokenizer, data_all, max_length, max_gen, prompt_format, args.dataset, device, model_name_or_path)

        masked_size = torch.zeros((32,8))
        for i in range(start_idx,n):
            masked_size[int(idxs[i]/8)][idxs[i]%8] = 8000
        model.model.config.given_adaptive_size = masked_size
        u_right = get_pred(model, tokenizer, data_all, max_length, max_gen, prompt_format, args.dataset, device, model_name_or_path)

        for i in range(n):
            if i < start_idx:
                cc_left[idxs[i]] += u_left-u_right
                count_left[idxs[i]] += 1
            else:
                cc_right[idxs[i]] += u_right-u_left
                count_right[idxs[i]] += 1

        
        if (_ + 1) % 5 == 0:
            data_to_save = {
                'cc_left': cc_left,
                'cc_right': cc_right,
                'count_left': count_left,
                'count_right': count_right,
                'sampling_count': _ + 1  
            }
            np.save(file_path, data_to_save)

    