#!/bin/bash
export CUDA_VISIBLE_DEVICES=7


pid=586043
while kill -0 $pid 2>/dev/null; do
    sleep 1
done
echo "进程 $pid 已结束"
declare -A model_steps
model_steps=(
    ["Qwen3-4B_grpo"]="1000"
    # ["Qwen3-8B_grpo"]="1000"
)

for MODEL in "${!model_steps[@]}"; do
    STEP="${model_steps[$MODEL]}"
    echo "Executing $MODEL at step $STEP"
    bash /data/pyzhu/a_wvs_eval/eval_batch.sh "$MODEL" "$STEP" 
done
