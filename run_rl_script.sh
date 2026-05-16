#!/bin/bash
#SBATCH --account=a0107
#SBATCH --job-name=cai_sft_training
#SBTACH --output=result.out
#SBATCH --error=results.err
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#SBATCH --time=06:00:00

uenv image pull pytorch/v2.9.1:v2
uenv start pytorch/v2.9.1:v2 --view=default

source /iopsstor/scratch/cscs/ajannali/venv/cai/bin/activate
cd /iopsstor/scratch/cscs/ajannali/project/causal_alignment

# Verify vllm is available before trying to background it
if ! command -v vllm &> /dev/null; then
    echo "ERROR: vllm not found in PATH. Is the venv activated?" >&2
fi

# Launch judge server on GPU 2-3 — log to file so errors are visible
# VLLM_USE_TRITON_FLASH_ATTN=0: flash_attn_2_cuda is not compiled for GH200 (sm_90a/aarch64);
# disabling triton flash attn makes vLLM fall back to native PyTorch attention for rotary embeddings.
echo "Starting judge server..."
# Qwen/Qwen2.5-72B-Instruct
CUDA_VISIBLE_DEVICES=3 vllm serve Qwen/Qwen3-8B \
    --port 8001 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.85 \
    --dtype bfloat16 > judge_server.log 2>&1 &

JUDGE_PID=$!
echo "Judge PID: $JUDGE_PID"

# Wait for the server — timeout after 10 minutes, check process is still alive each iteration
TIMEOUT=120   # 120 * 5s = 10 minutes
COUNT=0
until [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8001/health)" = "200" ]; do
    if ! kill -0 "$JUDGE_PID" 2>/dev/null; then
        echo "ERROR: Judge server process died. Last lines of judge_server.log:" >&2
        tail -20 judge_server.log >&2
        exit 1
    fi
    COUNT=$((COUNT + 1))
    if [ "$COUNT" -ge "$TIMEOUT" ]; then
        echo "ERROR: Judge server did not become ready after $((TIMEOUT * 5))s." >&2
        tail -20 judge_server.log >&2
        kill "$JUDGE_PID" 2>/dev/null
        exit 1
    fi
    echo "  waiting... (${COUNT}/${TIMEOUT})"
    sleep 5
done
echo "Judge server ready."

# Run policy training on GPU 0-1
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=3 ./src/training/train.py

kill $JUDGE_PID
