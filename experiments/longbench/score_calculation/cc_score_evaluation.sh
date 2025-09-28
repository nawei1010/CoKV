#!/bin/bash
MODEL="${MODEL:-models/Qwen3-32B}"
MAX_LEN=20000

echo "Testing $MODEL $DATASET $MAX_LEN"
DATASETS=("passage_count") 

GPUS=(4)
SET_SIZES=(256)
NUM_GPUS=${#GPUS[@]}  
BATCH_SIZE=$NUM_GPUS 
MODE="sv"

for ((j=0; j<$NUM_GPUS; j++)); do
    GPU_IDX=${GPUS[$j]}
    SET_SIZE=${SET_SIZES[$j]}
    DATASET=${DATASETS[$j]}
    scope=128

    export CUDA_VISIBLE_DEVICES=$GPU_IDX
    nohup python -u cc_score_evaluation.py \
        -m $MODEL \
        --max_length $MAX_LEN \
        --compress_args_path c"$scope"_w32_k7_maxpool.json \
        -d $DATASET \
        --gpu_id $GPU_IDX \
        --model 'qwen3' \
        --mode $MODE \
        --set_size $SET_SIZE \
        --floor_alpha 0.2 \
        --gqa_support \
        --sampling_number 288 &
done



