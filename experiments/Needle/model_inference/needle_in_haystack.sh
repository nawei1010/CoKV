#!/bin/bash

GPUS=(6)
NUM_GPUS=${#GPUS[@]}
MODE='cokv'
HEADKV_BETA=1.01
COKV_ALPHAS=(1)
MODEL_NAME='mistral'

for ((i=0; i<$NUM_GPUS; i++)); do
        GPU_IDX=${GPUS[$i]}
        COKV_ALPHA=${COKV_ALPHAS[$i]}
        scope=128
        export CUDA_VISIBLE_DEVICES=$GPU_IDX
        
        nohup python -u needle_in_haystack.py \
            --s_len 1000 \
            --e_len 31001 \
            --model_provider Mistral \
            --model_name "model_to_your_modedl/Mistral-7B-Instruct-v0.2" \
            --attn_implementation "flash_attention_2" \
            --step 1000 \
            --model_version "Mistral-7B-Instruct-v0.2" \
            --compress_args_path c"$scope"_w32_k7_maxpool.json \
            --mode $MODE \
            --floor_alpha 0.2 \
            --cokv_alpha $COKV_ALPHA \
            --model $MODEL_NAME \
            --headkv_beta $HEADKV_BETA \
            --gqa_support --pyram > "log"_"$MODE"_"$COKV_ALPHA.txt" 2>&1 &

done


