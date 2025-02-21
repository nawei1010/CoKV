#!/bin/bash
MODEL="path_to_your_model/Mistral-7B-Instruct-v0.2"
MAX_LEN=8000

echo "Testing $MODEL $DATASET $MAX_LEN"

#the datasets you want to compute ssv
DATASETS=("qasper")

GPUS=(0 1 2 3)
SET_SIZES=(32 64 96 128)
NUM_GPUS=${#GPUS[@]}  
BATCH_SIZE=$NUM_GPUS  
MODE="cokv"

for DATASET in "${DATASETS[@]}"; do
    echo "Processing dataset: $DATASET"
        for ((j=0; j<$NUM_GPUS; j++)); do
            GPU_IDX=${GPUS[$j]}
            SET_SIZE=${SET_SIZES[$j]}

            scope=128
            export CUDA_VISIBLE_DEVICES=$GPU_IDX
            nohup  python -u cal_global_sv.py \
                -m $MODEL \
                --max_length $MAX_LEN \
                --compress_args_path c"$scope"_w32_k7_maxpool.json \
                -d $DATASET \
                --gpu_id $GPU_IDX \
                --model 'mistral' \
                --mode $MODE \
                --set_size $SET_SIZE\
                --floor_alpha 0.2 \
                --gqa_support \
                --sampling_number 10000 &
        done
        wait
done


