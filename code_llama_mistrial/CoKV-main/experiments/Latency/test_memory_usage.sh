#!/bin/bash

modes=("base" "fix" "ada" "pyramid" "cokv" "headkv" "snapkv")
scopes=(64 128 256 512 1024)

for mode in "${modes[@]}"; do
    for scope in "${scopes[@]}"; do
        echo "Running with mode=$mode and cache_size=$scope"
        python test_memory_usage.py --mode "$mode" --cache_size "$scope"
        wait
    done
done
