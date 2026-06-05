#!/bin/bash
#SBATCH --account=a0107
#SBATCH --job-name=causal_verl_grpo
#SBATCH --output=/iopsstor/scratch/cscs/ajannali/project/verl_runs/verl_train_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/ajannali/project/verl_runs/verl_train_%j.err
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --partition=normal
#SBATCH --time=06:00:00
#SBATCH --environment=/iopsstor/scratch/cscs/ajannali/project/env_toml.toml

cd /iopsstor/scratch/cscs/ajannali/project/causal_alignment

# Set RESUME=true to resume from the latest checkpoint instead of starting fresh.
RESUME=true

rm -rf judge_server.log
rm -rf /iopsstor/scratch/cscs/ajannali/project/verl_runs
rm -rf /iopsstor/scratch/cscs/ajannali/project/causal_alignment/core_nid*

if [ "$RESUME" = "false" ]; then
    rm -rf /iopsstor/scratch/cscs/ajannali/project/causal_alignment/verl_training_test.log
    rm -rf /iopsstor/scratch/cscs/ajannali/project/causal_alignment/verl_metrics_test.csv
    rm -rf /iopsstor/scratch/cscs/ajannali/project/causal_alignment/src/output_RL/verl_checkpoints/*
fi

pip install --user \
  "transformers==4.55.4" \
  "tokenizers==0.21.1" \
  "huggingface_hub==0.34.0" \
  "numpy==1.26.4" \
  --force-reinstall

pip install --user matplotlib scipy scikit-learn linearmodels rdd

pip install --user verl==0.6.0 --force-reinstall --no-deps
pip install --user "uvloop<0.22"
pip install --user "torchdata==0.11.0" --no-deps

# ── Data preparation (runs once, rank-0 only via fcntl in the script) ──────
# echo "Running data preparation..."
# python3 /iopsstor/scratch/cscs/ajannali/project/causal_alignment/src/training/verl_/data_process.py
# [ $? -eq 0 ] || { echo "ERROR: data_process.py failed." >&2; kill "$JUDGE_PID"; }
# echo "Data preparation complete."

# ── Judge server (GPU 3) — thinking DISABLED, we just need 0/1 scoring ────
echo "Starting judge server on GPU 3..."
 
CUDA_VISIBLE_DEVICES=3 vllm serve Qwen/Qwen3-8B \
    --port 8001 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.18 \
    --max-model-len 1500 \
    --max-num-seqs 4 \
    --dtype bfloat16 \
    --override-generation-config '{"enable_thinking": false}' \
    > judge_server.log 2>&1 &
 
JUDGE_PID=$!
echo "Judge server PID: $JUDGE_PID"
 
TIMEOUT=120
COUNT=0
until [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8001/health)" = "200" ]; do
    if ! kill -0 "$JUDGE_PID" 2>/dev/null; then
        echo "ERROR: Judge server died." >&2
        tail -20 judge_server.log >&2
    fi
    COUNT=$((COUNT + 1))
    [ "$COUNT" -ge "$TIMEOUT" ] && {
        echo "ERROR: Judge timeout after $((TIMEOUT * 5))s." >&2
        tail -20 judge_server.log >&2
        kill "$JUDGE_PID"
    }
    echo "  waiting for judge... (${COUNT}/${TIMEOUT})"
    sleep 5
done
echo "Judge server ready."
 
# ── Ray cluster (all 4 GPUs for training) ──────────────────────────────────
unset ROCR_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0,1,2,3
export RAY_USAGE_STATS_ENABLED=0
 
ray start --head \
    --num-cpus=48 \
    --num-gpus=4 \
    --port=6379 \
    --dashboard-port=8265 \
    --block &
 
RAY_HEAD_PID=$!
sleep 15
echo "Ray cluster started."
 
# ── Training step/freq math ─────────────────────────────────────────────────
# total_steps = epochs * ceil(train_size / train_batch_size)
TRAIN_SIZE=$(python3 -c "
import math, pandas as pd
df = pd.read_parquet('src/output_RL/train.parquet')
train_size = len(df)
steps = 3 * math.ceil(train_size / 20)
print(train_size, steps)
" | tee /dev/stderr | tail -1)
 
# parse both values
TRAIN_SIZE_N=$(echo $TRAIN_SIZE | awk '{print $1}')
TOTAL_STEPS=$(echo $TRAIN_SIZE | awk '{print $2}')
TEST_FREQ=$(python3 -c "print(max(15, $TOTAL_STEPS // 100))")
echo "train_size=$TRAIN_SIZE_N  total_steps=$TOTAL_STEPS  TEST_FREQ=$TEST_FREQ"
 
# ── Resume logic ────────────────────────────────────────────────────────────
CKPT_DIR=/iopsstor/scratch/cscs/ajannali/project/causal_alignment/src/output_RL/verl_checkpoints
LATEST_CKPT=$(find "$CKPT_DIR" -maxdepth 1 -type d -name "global_step_*" | sort -V | tail -n 1)
 
RESUME_ARG=""
if [ "${RESUME:-}" = "true" ] && [ -n "$LATEST_CKPT" ]; then
    RESUME_ARG="trainer.resume_mode=resume_path trainer.resume_from_path=$LATEST_CKPT"
    echo "Resuming from: $LATEST_CKPT"
fi
 
# ── veRL GRPO training ──────────────────────────────────────────────────────
# Memory budget per GPU (H200 95 GB):
#   Actor FSDP shard (16 GB / 4):          ~4.1 GB
#   Ref FSDP shard (16 GB / 4):            ~4.0 GB   (no offload — we have room)
#   Optimizer state (Adam fp32, /4):       ~12.0 GB
#   Activations (micro=4, seq=4000):       ~4.0 GB
#   vLLM weights (full copy, TP=1):        ~16.0 GB
#   vLLM KV cache (util=0.45):            ~31.0 GB
#   Judge on GPU3 (weights+KV @ 0.2):     ~32.0 GB  (only GPU 3)
#   ─────────────────────────────────────────────────
#   GPU 0-2 total:                         ~71 GB / 95 GB
#   GPU 3 total:                           ~71 + 32 - 4 (ref, ref shard is there too) ≈ tight but ok
#
# Batch math:
#   train_batch_size=20, rollout.n=6 → 120 total trajectories per iter
#   ppo_mini_batch_size=20           → 5 optimizer steps per iter  (120/20)
#   ppo_micro_batch_size_per_gpu=5   → 1 grad accum step per mini  (20/(5*4GPUs) = 1.0 → ceil=1)
#   log_prob_micro_batch_size_per_gpu=6 → forward only, doubled for throughput
 
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    \
    data.train_files=src/output_RL/train.parquet \
    data.val_files=src/output_RL/test.parquet \
    data.train_batch_size=20 \
    data.max_prompt_length=2250 \
    data.max_response_length=2250 \
    data.truncation=left \
    data.dataloader_num_workers=4 \
    data.shuffle=True \
    +data.apply_chat_template_kwargs.enable_thinking=true \
    +data.apply_chat_template_kwargs.thinking_budget=650 \
    \
    actor_rollout_ref.model.path=Qwen/Qwen3-8B \
    actor_rollout_ref.model.lora_rank=16 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.target_modules=all-linear \
    ++actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    +actor_rollout_ref.model.override_config.torch_dtype=bfloat16 \
    +actor_rollout_ref.model.override_config.attn_implementation=flash_attention_3 \
    \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.05 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.optim.lr=2e-5 \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    \
    actor_rollout_ref.actor.ppo_mini_batch_size=20 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=5 \
    \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    \
    actor_rollout_ref.rollout.n=6 \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.top_k=10 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=6 \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.max_model_len=4500 \
    \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=6 \
    \
    reward_model.enable=False \
    custom_reward_function.path=/iopsstor/scratch/cscs/ajannali/project/causal_alignment/src/training/verl_/reward.py \
    custom_reward_function.name=compute_score \
    \
    trainer.critic_warmup=0 \
    $RESUME_ARG \
    trainer.logger='["console", "file"]' \
    trainer.project_name=causal_alignment \
    trainer.experiment_name=qwen3_8b_grpo \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=$TEST_FREQ \
    trainer.total_epochs=3 \
    trainer.default_local_dir=$CKPT_DIR \
    2>&1 | tee -a verl_training_test.log

TRAIN_EXIT=${PIPESTATUS[0]}

# ── Parse logs → CSV (copy verl_metrics.csv to laptop for plotting) ─────────
echo "Parsing training log..."
python3 /iopsstor/scratch/cscs/ajannali/project/causal_alignment/src/training/verl_/parse_verl_logs.py verl_training_test.log verl_metrics_test.csv

# ── Cleanup ─────────────────────────────────────────────────────────────────
ray stop
kill "$JUDGE_PID" 2>/dev/null
# wait "$RAY_HEAD_PID" 2>/dev/null

echo "Training exit code: $TRAIN_EXIT"
exit $TRAIN_EXIT 