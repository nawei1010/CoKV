#!/usr/bin/env bash


GPU_IDS="5"   
export CUDA_VISIBLE_DEVICES=${GPU_IDS}
export PYTHONUNBUFFERED=1  

model_name="${MODEL:-models/Qwen3-32B}"
ctx_len_min=1000
ctx_len_max=32000
reg_weight=0.05
lr=0.002
num_passkey=10

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${ROOT_DIR}" 

[ -z "${PYTHONPATH+x}" ] && PYTHONPATH=""
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH}"

model_tag=$(basename "${model_name}")
exp_name="${model_tag}-lr=${lr}-reg=${reg_weight}-ctx=${ctx_len_min}_${ctx_len_max}-p=${num_passkey}"
out_dir="attn_patterns/${exp_name}"
mkdir -p "${out_dir}"

GPU_COUNT=$(python -c 'import torch;print(torch.cuda.device_count())' 2>/dev/null || echo 1)
echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} => ${GPU_COUNT} logical GPUs"

torchrun --nnodes 1 --nproc_per_node ${GPU_COUNT} \
    "${ROOT_DIR}/duo_attn/train.py" \
    --model_name "${model_name}" \
    --batch_size 1 \
    --max_length "${ctx_len_max}" \
    --dataset_name "${DATASET_PATH:-datasets/booksum.jsonl.zst}" \
    --sink_size 128 \
    --recent_size 256 \
    --num_steps 2000 \
    --lr "${lr}" \
    --reg_weight "${reg_weight}" \
    --exp_name "${exp_name}" \
    --min_needle_depth_ratio 0.05 \
    --max_needle_depth_ratio 0.95 \
    --context_length_min "${ctx_len_min}" \
    --context_length_max "${ctx_len_max}" \
    --context_lengths_num_intervals 50 \
    --depth_ratio_num_intervals 1000 \
    --gradient_accumulation_steps 1 \
    --num_passkey "${num_passkey}" \
    --dataset_format multiple_passkey \
    --output_dir "${out_dir}"

echo "finish ${exp_name} output_dir: ${out_dir}"

