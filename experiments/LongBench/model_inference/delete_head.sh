#!/bin/bash
MAX_LEN=8000
MODES=("cokv")
DEL_GROUP=16
FLAGS=('low')
MODEL_NAME='mistral'
GPUS=(6)  
MODEL="path_to_your_model/Mistral-7B-Instruct-v0.2"
#MODEL="path_to_your_model/Llama-3-8B-Instruct"
DATASET="path_to_your_project/CoKV/assets/datasets/LongBench-test"

echo "Testing $MODEL $DATASET $MAX_LEN"

NUM_GPUS=${#GPUS[@]} 
NUM_FLAGS=${#FLAGS[@]}
for ((k=0; k <$NUM_FLAGS; k+=1));do
    FLAG=${FLAGS[$k]}
    for ((i=0; i<$NUM_GPUS; i+=1)); do
        GPU_IDX=${GPUS[$i]}
        MODE=${MODES[$i]}
        DEL_GROUP=$DEL_GROUP
        MODEL_NAME=$MODEL_NAME
        MODEL=$MODEL
        scope=128
        CUDA_VISIBLE_DEVICES=$GPU_IDX \
        nohup python delete_head.py \
        -m $MODEL \
        --max_length $MAX_LEN \
        -d $DATASET \
        --mode $MODE \
        --compress_args_path c"$scope"_w32_k7_maxpool.json \
        --floor_alpha 0.2 \
        --model $MODEL_NAME \
        --flag $FLAG\
        --del_group $DEL_GROUP\
        --out_name "$MODEL_NAME"_"test_delete_head"_"floor_size"_"$MODE"_"$DEL_GROUP"_"$FLAG" \
        --gqa_support & 
    done
    wait
done

#nohup ./delete_head.sh > pred4.log 2>&1 &

