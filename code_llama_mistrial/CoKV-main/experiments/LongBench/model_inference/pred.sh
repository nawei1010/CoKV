#!/bin/bash
MAX_LEN=8000
MODE="cokv"
SCOPES=(64 128 256 512 1024)
MODEL_NAME='mistral'
GPUS=(0 1 2 3 4 5 6)  
SIZES=(1 5 10 15 20 30 40)
BETAS=(1.01 1.01 1.01 1.01 1.01 1.01 1.01)
MODEL="path_to_your_model/Mistral-7B-Instruct-v0.2"
#MODEL="path_to_your_model/Llama-3-8B-Instruct"
DATASET="path_to_your_project/CoKV/assets/datasets/LongBench-test"

echo "Testing $MODEL $DATASET $MAX_LEN"

NUM_GPUS=${#GPUS[@]}  
NUM_SCOPES=${#SCOPES[@]}

for ((j=0; j<$NUM_SCOPES; j+=1)); do
  scope=${SCOPES[$j]}
  echo "Processing scope=$scope"

  for ((i=0; i<$NUM_GPUS; i+=1)); do
    GPU_IDX=${GPUS[$i]}
    SIZE=${SIZES[$i]}
    BETA=${BETAS[$i]}

    CUDA_VISIBLE_DEVICES=$GPU_IDX \
    nohup python pred.py \
      -m $MODEL \
      --max_length $MAX_LEN \
      -d $DATASET \
      --mode $MODE \
      --compress_args_path c"$scope"_w32_k7_maxpool.json \
      --floor_alpha 0.2 \
      --sv_beta $SIZE \
      --headkv_beta $BETA \
      --model $MODEL_NAME \
      --out_name "$MODEL_NAME"_"test_all"_"floor_size_$SIZE"_"$MODE"_"$scope" \
      --gqa_support &  
  done

  wait  
done
#nohup ./pred.sh > pred.log 2>&1 &