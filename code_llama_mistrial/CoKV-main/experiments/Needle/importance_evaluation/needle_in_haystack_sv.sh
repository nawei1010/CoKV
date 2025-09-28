#!/bin/bash

GPUS=(0 1 2 3)
NUM_GPUS=${#GPUS[@]}  
BATCH_SIZE=$NUM_GPUS 
MODE="cokv"
SET_SIZES=(32 64 96 128)

for ((i=0; i<$NUM_GPUS; i++)); do
        GPU_IDX=${GPUS[$i]}
        SET_SIZE=${SET_SIZES[$i]}
        scope=128
        export CUDA_VISIBLE_DEVICES=$GPU_IDX
        
        nohup python -u needle_in_haystack_sv.py \
            --s_len 1000 \
            --e_len 31001 \
            --model_provider Mistral \
            --model_name "path_to_your_model/Mistral-7B-Instruct-v0.2" \
            --attn_implementation "flash_attention_2" \
            --step 2500 \
            --model_version "Mistral-7B-Instruct-v0.2" \
            --compress_args_path c"$scope"_w32_k7_maxpool.json \
            --gpu_idx $GPU_IDX\
            --mode $MODE \
            --floor_alpha 0.2 \
            --gqa_support \
            --set_size $SET_SIZE\
            --sampling_number 10000 > "log_MC_$GPU_IDX.txt" 2>&1 &
done


