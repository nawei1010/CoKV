#!/bin/bash
MAX_LEN=20000
MODE="ada"
SCOPES=(128)
GPUS=(5)  
SIZES=(10)
BETAS=(1.01)
MODEL="${MODEL:-models/Qwen3-32B}"
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
    nohup python inference.py \
      -m $MODEL \
      --max_length $MAX_LEN \
      --mode $MODE \
      --compress_args_path c"$scope"_w32_k7_maxpool.json \
      --floor_alpha 0.2 \
      --sv_beta $SIZE \
      --headkv_beta $BETA \
      --out_dir "$MODEL_NAME"_"floor_size_$SIZE"_"$MODE"_"$scope" \
      --gqa_support > pred_snapkv_"$SIZE".log 2>&1 &  
  done

  wait 
done
#nohup ./pred.sh > pred.log 2>&1 &