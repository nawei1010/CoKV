#!/bin/bash
set -euo pipefail

MODE='ada'
HEADKV_BETA=1.01
MODEL_NAME='qwen3'
MODEL_PATH="/path/to/models/Qwen3-32B"  # Update this path to your model
HAYSTACK_DIR="./adaptive_kv/assets/datasets/PaulGrahamEssays"

cd ./experiments/needle


COKV_ALPHA=20
scope=128
export CUDA_VISIBLE_DEVICES=5

nohup python -u needle_inference.py \
  --s_len 1000 \
  --e_len 128001 \
  --model_provider Qwen3 \
  --model_name "$MODEL_PATH" \
  --haystack_dir "$HAYSTACK_DIR" \
  --attn_implementation "flash_attention_2" \
  --step 4000 \
  --model_version "Qwen3-32B" \
  --scope "$scope" \
  --compress_args_path c"$scope"_w32_k7_maxpool.json \
  --mode $MODE \
  --floor_alpha 0.2 \
  --cokv_alpha $COKV_ALPHA \
  --model $MODEL_NAME \
  --headkv_beta $HEADKV_BETA \
  --gqa_support --pyram > pred_$MODE"_"$COKV_ALPHA.log 2>&1 &
