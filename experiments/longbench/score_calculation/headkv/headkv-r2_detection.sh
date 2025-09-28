CUDA_VISIBLE_DEVICES='0,1,2' nohup python headkv-r2_detection.py  \
    --model_path ${MODEL:-models/Qwen3-32B} \
    --model_provider QWEN3 \
    --s 0 \
    --e 10000 \
    --task retrieval_reasoning_heads \
    --haystack_dir ${HAYSTACK_DIR:-experiments/longbench/score_calculation/headkv/haystack_for_detect_r2} \

   
    