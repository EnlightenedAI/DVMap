#!/bin/bash
set -e

MODEL=$1
STEP=$2

# 打印看看参数是否正确
echo "Running model: $MODEL"
echo "Step: $STEP"

DO_TRANS=true
VLLM_TASK=true

MODEL_NAME="${MODEL}-global_step_${STEP}"

CHECKPOINTS_PATH="PATH/${MODEL}/global_step_${STEP}/actor"
SAVED_MODEL_PATH="PATH/${MODEL_NAME}"

PORT="5456"
OUTPUT_CSV_NAME="${MODEL_NAME}_T0.7_nocot.csv"

EVAL_DATA="Triple-Generalization_Evaluation_Benchmark/Cross-Demographic"
EVAL_DATA_extrap="Triple-Generalization_Evaluation_Benchmark/Cross-Value"
EVAL_DATA_country="Triple-Generalization_Evaluation_Benchmark/Cross-Country"
OUTPUT_DIR="/data/pyzhu/a_wvs_eval/output/answer/${OUTPUT_CSV_NAME}"
OUTPUT_DIR_extrap="/data/pyzhu/a_wvs_eval/output/extrap_answer/${OUTPUT_CSV_NAME}"
OUTPUT_DIR_country="/data/pyzhu/a_wvs_eval/output/answer_guojia/${OUTPUT_CSV_NAME}"
BASE_URL="http://localhost:${PORT}/v1"
API_KEY="sk-LOCAL-TEST-KEY"

echo "[INFO] CHECKPOINTS_PATH = $CHECKPOINTS_PATH"
echo "[INFO] SAVED_MODEL_PATH = $SAVED_MODEL_PATH"
echo "[INFO] BASE_URL = $BASE_URL"
echo "[INFO] OUTPUT_DIR = $OUTPUT_DIR"


if [ "$DO_TRANS" = "true" ]; then
    echo "[INFO] Running trans.py..."
    python /data/pyzhu/a_wvs_eval/trans.py \
        "$CHECKPOINTS_PATH" "$SAVED_MODEL_PATH"
    echo "[INFO] Conversion done."
else
    echo "[INFO] Skipping trans.py..."
fi


if [ "$VLLM_TASK" = "true" ]; then
    echo "[INFO] Starting vLLM Server..."
    nohup vllm serve $SAVED_MODEL_PATH \
        --trust-remote-code \
        --tensor-parallel-size 1 \
        --port $PORT \
        --gpu-memory-utilization 0.7\
        > vllm_task.log 2>&1 &

    VLLM_PID=$!
    echo "[INFO] vLLM PID: $VLLM_PID"

    echo "[INFO] Waiting for vLLM to load model..."

    MAX_WAIT=1000       
    CHECK_INTERVAL=2
    ELAPSED=0

    # 日志关键词
    SUCCESS_KEYWORD="Uvicorn running"
    ALT_SUCCESS="Application startup complete"

    while true; do
        if ! ps -p $VLLM_PID > /dev/null; then
            echo "[ERROR] vLLM crashed during startup! Check vllm_task.log."
            exit 1
        fi

        # 2. 检查日志关键词（最准确）
        if grep -q "$SUCCESS_KEYWORD" vllm_task.log || grep -q "$ALT_SUCCESS" vllm_task.log; then
            echo "[INFO] vLLM started successfully (log detected)."
            break
        fi

        # 3. 超时
        if [ $ELAPSED -ge $MAX_WAIT ]; then
            echo "[ERROR] vLLM failed to start within $MAX_WAIT seconds!"
            echo "[ERROR] Check vllm_task.log for details."
            kill $VLLM_PID 2>/dev/null || true
            exit 1
        fi

        sleep $CHECK_INTERVAL
        ELAPSED=$((ELAPSED + CHECK_INTERVAL))
    done
fi


echo "[INFO] Running evaluation..."

python /data/pyzhu/a_wvs_eval/eval_api_parallel.py \
    --model_name "$SAVED_MODEL_PATH" \
    --eval_data_path "$EVAL_DATA" \
    --output_csv_path "$OUTPUT_DIR" \
    --base_url "$BASE_URL" \
    --api_key "$API_KEY" \
    --temperature 0.7 \
    --batch_size 32 \
    --max_workers 32 \
    --max_retries 5 \
    --resume




python /data/pyzhu/a_wvs_eval/eval_api_parallel.py \
    --model_name "$SAVED_MODEL_PATH" \
    --eval_data_path "$EVAL_DATA_extrap" \
    --output_csv_path "$OUTPUT_DIR_extrap" \
    --base_url "$BASE_URL" \
    --api_key "$API_KEY" \
    --temperature 0.7 \
    --batch_size 32 \
    --max_workers 32 \
    --max_retries 5 \
    --resume

python /data/pyzhu/a_wvs_eval/eval_api_parallel.py \
    --model_name "$SAVED_MODEL_PATH" \
    --eval_data_path "$EVAL_DATA_country" \
    --output_csv_path "$OUTPUT_DIR_country" \
    --base_url "$BASE_URL" \
    --api_key "$API_KEY" \
    --temperature 0.7 \
    --batch_size 32 \
    --max_workers 32 \
    --max_retries 5 \
    --resume


if [ "$VLLM_TASK" = "true" ]; then
    echo "[INFO] Stopping vLLM..."
    kill $VLLM_PID
    while lsof -i:$PORT >/dev/null 2>&1; do
    sleep 1
    done
fi


echo "[INFO] Pipeline completed successfully!"

rm -r $SAVED_MODEL_PATH

