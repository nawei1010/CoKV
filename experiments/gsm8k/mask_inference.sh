#!/bin/bash
MAX_LEN=20000
MODE="sv"
SCOPES=(128)
MODEL_NAME='qwen3'
GPUS=(5)  
SIZES=(10)
FLAG='large'
MASK_NUMBERS=(16 32 64 96) 
BETAS=(1.01 1.01 1.01 1.01)
MODEL="${MODEL:-models/Qwen3-32B}"


echo "Testing $MODEL $DATASET $MAX_LEN"

NUM_GPUS=${#GPUS[@]} 
NUM_MASKS=${#MASK_NUMBERS[@]}
scope=128
for ((j=0; j<$NUM_MASKS; j+=1)); do
  
  echo "Processing scope=$scope"
  
  for ((i=0; i<$NUM_GPUS; i+=1)); do
    GPU_IDX=${GPUS[$i]}
    SIZE=${SIZES[$i]}
    BETA=${BETAS[$i]}
    MASK_NUMBER=${MASK_NUMBERS[$j]}

    
    CUDA_VISIBLE_DEVICES=$GPU_IDX \
    nohup python mask_inference.py \
      -m $MODEL \
      --max_length $MAX_LEN \
      --mode $MODE \
      --flag $FLAG \
      --mask_number $MASK_NUMBER \
      --compress_args_path c"$scope"_w32_k7_maxpool.json \
      --floor_alpha 0.2 \
      --sv_beta $SIZE \
      --headkv_beta $BETA \
      --model $MODEL_NAME \
      --gqa_support > pred_mask_sv_"$MASK_NUMBER".log 2>&1 &  

  wait 
done
#nohup ./pred.sh > pred.log 2>&1 &