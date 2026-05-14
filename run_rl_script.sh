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

# Launch judge server on GPU 2-3
CUDA_VISIBLE_DEVICES=2,3 vllm serve Qwen/Qwen2.5-72B-Instruct \
    --port 8001 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.85 \
    --dtype bfloat16 &

JUDGE_PID=$!

echo "Waiting for judge server to be ready..."
until curl -sf http://localhost:8001/health > /dev/null 2>&1; do
    sleep 5
done
echo "Judge server ready."

# Run policy training on GPU 0-1
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 ./src/training/train.py

kill $JUDGE_PID
