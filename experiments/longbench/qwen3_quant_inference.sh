#!/bin/bash
MAX_LEN=20000
MODE="sv"
MODEL_NAME='qwen3'
GPUS=(4)  
HBIT=8
LBIT=4
MODEL="${MODEL:-models/Qwen3-32B}"
DATASET="${DATASET:-datasets/LongBench-test}"

echo "Testing $MODEL $DATASET $MAX_LEN"

NUM_GPUS=${#GPUS[@]}  

for ((j=0; j<1; j+=1)); do
  scope=128
  echo "Processing quantization=$HBIT $LBIT"

  for ((i=0; i<$NUM_GPUS; i+=1)); do
    GPU_IDX=${GPUS[$i]}
   

  
    CUDA_VISIBLE_DEVICES=$GPU_IDX \
    nohup python qwen3_quant_inference.py \
      -m $MODEL \
      --max_length $MAX_LEN \
      -d $DATASET \
      --mode $MODE \
      --high_bit $HBIT \
      --low_bit $LBIT \
      --compress_args_path c"$scope"_w32_k7_maxpool.json \
      --floor_alpha 0.2 \
      --model $MODEL_NAME \
      --out_name "$MODEL_NAME"_"quantization_test_$HBIT_$LBIT"_"$MODE"_"$scope" \
      --gqa_support > quantization_sv.log 2>&1 &  
  done

  wait  
done
#nohup ./pred.sh > pred.log 2>&1 &