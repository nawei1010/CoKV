#!/bin/bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
MODES=("base")
SCOPES=(128)

cd ./experiments/memory_latency

for mode in "${MODES[@]}"; do
  for scope in "${SCOPES[@]}"; do
    echo "Running memory usage with mode=$mode cache_size=$scope"
    CUDA_VISIBLE_DEVICES=0,1 python memory.py --mode "$mode" --cache_size "$scope" | cat
    wait
  done
done