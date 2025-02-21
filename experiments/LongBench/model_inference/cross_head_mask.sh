#!/bin/bash
MAX_LEN=8000
MODE="cokv"
DEL_GROUPS=(96)
FLAGS=('low' 'top')
MODEL_NAME='llama'
SOURCE_DATASET='lcc'
TARGET_DATASET='repobench-p'
GPU_IDX=4
#MODEL="path_to_your_model/Mistral-7B-Instruct-v0.2"
MODEL="path_to_your_model/Llama-3-8B-Instruct"
DATASET="path_to_your_project/CoKV/assets/datasets/LongBench-test"

echo "Testing $MODEL $DATASET $MAX_LEN"

NUM_FLAGS=${#FLAGS[@]}
NUM_DEL_GROUPS=${#DEL_GROUPS[@]}

for ((i=0; i<$NUM_DEL_GROUPS; i+=1)); do
    for ((k=0; k <$NUM_FLAGS; k+=1));do
        FLAG=${FLAGS[$k]}
        DEL_GROUP=${DEL_GROUPS[$i]}
        scope=128
        TORCH_USE_CUDA_DSA=1 CUDA_VISIBLE_DEVICES=$GPU_IDX \
        nohup python cross_head_mask.py \
        -m $MODEL \
        --max_length $MAX_LEN \
        -d $DATASET \
        --mode $MODE \
        --compress_args_path c"$scope"_w32_k7_maxpool.json \
        --floor_alpha 0.2 \
        --source_dataset $SOURCE_DATASET \
        --target_dataset $TARGET_DATASET \
        --model $MODEL_NAME \
        --flag $FLAG\
        --del_group $DEL_GROUP\
        --out_name "$MODEL_NAME"_"test_delete_head"_"$SOURCE_DATASET"_"$TARGET_DATASET"_"$MODE"_"$DEL_GROUP"_"$FLAG" \
        --gqa_support &  
        wait
    done
done
#nohup ./cross_head_mask.sh > pred4.log 2>&1 &

