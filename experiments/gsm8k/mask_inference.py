import sys
import os
# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)
import json
import gc
import time
import random
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from adaptive_kv.monkeypatch.monkeypatch import replace_qwen_adaptive
from adaptive_kv.monkeypatch.utils import cc_mask, headkv_mask, random_mask,cc_mask_average

from experiments.gsm8k.data_utils import prepare_gsm8k_dataset
from experiments.gsm8k.metrics import exact_match_numeric


def seed_everything(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(path: str):
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
        local_files_only=True,
    )
    model = model.eval()
    return model, tokenizer


def config_compress(model,
                   window_size=32,
                   base_capacity=512,
                   kernel_size=7,
                   pooling="maxpool",
                   floor_alpha=0.5,
                   pyram_mode=False,
                   pyram_beta=20,
                   normalize=True,
                   skip=0,
                   gqa_support=False,
                   given_adaptive_size=None,
                   mask_heads=None):
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


def build_prompt(question: str) -> str:
    return (
        "Task: Solve the following math word problem. You may reason briefly, then give ONLY the final numeric answer.\n"
        "Rules:\n"
        "- Do NOT include units or extra words\n"
        "- The LAST line must be exactly: Final Answer: <number>\n\n"
        f"Problem: {question}\n"
        "Answer (end with 'Final Answer: <number>'):\n"
    )


def build_5shot_prompt(question: str, shots: list) -> str:
    prefix = (
        "Task: Solve math word problems. For each example, show concise Solution and finish with 'Final Answer: <number>'.\n"
        "Rules:\n"
        "- Do NOT include units or extra words\n"
        "- Always end with: Final Answer: <number>\n\n"
    )
    examples = []
    for s in shots:
        q = s.get("question", "")
        rationale = s.get("rationale", "")
        a = s.get("answers", [""])[0]
        examples.append(f"Problem: {q}\nSolution: {rationale}\nFinal Answer: {a}\n\n")
    examples_text = "".join(examples)
    return (
        prefix
        + examples_text
        + f"Now solve the next problem.\nProblem: {question}\n"
        + "Answer (end with 'Final Answer: <number>'):\n"
    )


def predict_and_score(model, tokenizer, dataset_rows, max_length: int, max_gen: int):
    device = "cuda"
    scores = []
    for ex in tqdm(dataset_rows):

        prompt = ex.get("prompt", None)
        if not prompt:
            prompt = build_prompt(ex["question"]) if "question" in ex else ""
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False  # Official parameter: enable/disable thinking mode
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        context_length = model_inputs.input_ids.shape[-1]
        output = model.generate(
            **model_inputs,
            max_new_tokens=max_gen,
            num_beams=1,
            do_sample=False,
            temperature=1.0,
            eos_token_id=[tokenizer.eos_token_id],
        )[0]
        output_ids = output[context_length:].tolist()

        def rfind_subsequence(lst, sub):
            if not sub:
                return None
            for i in range(len(lst) - len(sub), -1, -1):
                if lst[i:i+len(sub)] == sub:
                    return i + len(sub)
            return None

        end_think_seq = tokenizer.encode("</think>", add_special_tokens=False)
        idx = rfind_subsequence(output_ids, end_think_seq)
        if idx is None:
            idx = 0
        thinking_content = tokenizer.decode(output_ids[:idx], skip_special_tokens=True).strip("\n")
        content = tokenizer.decode(output_ids[idx:], skip_special_tokens=True).strip("\n")

        pred = content if content else tokenizer.decode(output[context_length:], skip_special_tokens=True)
        gold_list = ex.get("answers", [])
        score = 0.0
        
        for g in gold_list:
            score = max(score, exact_match_numeric(pred, g))
        scores.append(score)
        gc.collect(); torch.cuda.empty_cache()
    return float(np.mean(scores)) if scores else 0.0


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name_or_path', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--max_length', type=int, default=8000)
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--compress_args_path', type=str, default=None)
    parser.add_argument('--mode', type=str, choices=['ada', 'fix', 'base', 'sv', 'headkv', 'snapkv','random'], default='base')
    parser.add_argument('--gqa_support', action='store_true', default=False)
    parser.add_argument('--floor_alpha', type=float, default=0.2)
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument("--flag",type=str, default='large', help="mask top or bottom")
    parser.add_argument("--mask_number",type=int, default=0, help="mask heads number")
    parser.add_argument('--pyram', action='store_true')
    parser.add_argument('--pyram_beta', type=int, default=20)
    parser.add_argument('--skip', type=int, default=0)
    parser.add_argument('--sv_beta', type=int)
    parser.add_argument('--headkv_beta', type=float)
    parser.add_argument('--out_dir', type=str, default='./experiments/gsm8k/pred')
    parser.add_argument('--fewshot_k', type=int, default=5)
    args = parser.parse_args()

    seed_everything(42)
    print("args.model_name_or_path: ", args.model_name_or_path)
    # adaptive replace if needed
    compress_args = {}
    if args.compress_args_path:
        compress_args = json.load(open(args.compress_args_path, 'r')) if os.path.isabs(args.compress_args_path) \
            else json.load(open(os.path.join('./experiments/longbench/config', args.compress_args_path), 'r'))
        compress_args['window_size'] = 8
        compress_args['floor_alpha'] = args.floor_alpha
        compress_args['gqa_support'] = args.gqa_support
        compress_args['normalize'] = args.normalize
        compress_args['pyram_mode'] = args.pyram
        compress_args['skip'] = args.skip
        compress_args['pyram_beta'] = args.pyram_beta
        compress_args['mask_heads'] = None
        if args.mode in ['ada', 'sv', 'random', 'headkv', 'snapkv']:
            replace_qwen_adaptive()
    print("args.model_name_or_path: ", args.model_name_or_path)
    model, tokenizer = load_model_and_tokenizer(args.model_name_or_path)
    if args.compress_args_path:
        model = config_compress(model, **compress_args)

    # set given head sizes if requested
    if args.compress_args_path:
        if args.mode == 'sv':
            model.model.config.given_adaptive_size = cc_mask_average(args.model, args.flag, args.mask_number)
        elif args.mode == 'headkv':
            model.model.config.given_adaptive_size = headkv_mask(args.model,  args.flag, args.mask_number)
        elif args.mode == 'random':
            model.model.config.given_adaptive_size = random_mask(args.mask_number)
       

    # Prepare dataset into assets/datasets
    paths = prepare_gsm8k_dataset('./adaptive_kv/assets/datasets', valid_size=50)
    train_rows = [json.loads(line) for line in open(paths['train'], 'r', encoding='utf-8')]
    test_rows = [json.loads(line) for line in open(paths['test'], 'r', encoding='utf-8')]

    # sample k shots from train
    k = max(0, int(args.fewshot_k))
    shots = train_rows[:k] if len(train_rows) >= k else train_rows

    # wrap prompts
    fs_test_rows = []
    for ex in test_rows:
        fs_test_rows.append({
            "question": ex["question"],
            "answers": ex["answers"],
            "prompt": build_5shot_prompt(ex["question"], shots),
            "all_classes": ex.get("all_classes", []),
            "length": ex.get("length", len(ex["question"]))
        })

    acc = predict_and_score(model, tokenizer, fs_test_rows, args.max_length, args.max_new_tokens)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'results.json'), 'w') as f:
        json.dump({"accuracy": acc}, f)
    print(f"Test accuracy: {acc:.4f}")


if __name__ == '__main__':
    main()


