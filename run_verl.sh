#!/bin/bash
#SBATCH --account=a0107
#SBATCH --job-name=causal_verl_grpo
#SBATCH --output=verl_train.out
#SBATCH --error=verl_train.err
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --partition=normal
#SBATCH --time=06:00:00
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default

# uenv image pull pytorch/v2.9.1:v2
uenv start pytorch/v2.9.1:v2 --view=default

source /iopsstor/scratch/cscs/ajannali/venv/cai/bin/activate
cd /iopsstor/scratch/cscs/ajannali/project/causal_alignment

# ── Data preparation (runs once, rank-0 only via fcntl in the script) ──────
echo "Running data preparation..."
python3 src/training/verl_/data_process.py
[ $? -eq 0 ] || { echo "ERROR: data_process.py failed." >&2; kill "$JUDGE_PID"; }
echo "Data preparation complete."

# ── Judge server (GPU 3) — thinking DISABLED, we just need 0/1 scoring ────
echo "Starting judge server on GPU 3..."

# for some reason, I can't vllm and verl to sit together!

# CUDA_VISIBLE_DEVICES=3 vllm serve Qwen/Qwen3-8B \
#     --port 8001 \
#     --tensor-parallel-size 1 \
#     --gpu-memory-utilization 0.4 \
#     --dtype bfloat16 \
#     --override-generation-config '{"enable_thinking": false}' \
#     > judge_server.log 2>&1 &

# export VLLM_ATTENTION_BACKEND=TORCH_SDPA
# export TRANSFORMERS_NO_FLASH_ATTN=1
# export TRANSFORMERS_ATTN_IMPLEMENTATION=sdpa

CUDA_VISIBLE_DEVICES=3 python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --port 8001 \
    --tp 1 \
    --mem-fraction-static 0.4 \
    --dtype bfloat16 \
    --attention-backend torch_native \
    > judge_server.log 2>&1 &

JUDGE_PID=$!
echo "Judge PID: $JUDGE_PID"

# Wait for judge — timeout after 10 minutes
TIMEOUT=120
COUNT=0
until [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8001/health)" = "200" ]; do
    if ! kill -0 "$JUDGE_PID" 2>/dev/null; then
        echo "ERROR: Judge server died." >&2
        tail -20 judge_server.log >&2
        # exit 1
    fi
    COUNT=$((COUNT + 1))
    [ "$COUNT" -ge "$TIMEOUT" ] && {
        echo "ERROR: Judge timeout after $((TIMEOUT * 5))s." >&2
        tail -20 judge_server.log >&2
        kill "$JUDGE_PID"
        # exit 1
    }
    echo "  waiting for judge... (${COUNT}/${TIMEOUT})"
    sleep 5
done
echo "Judge server ready."

# ── Ray cluster (GPUs 0-2 for training) ────────────────────────────────────
export CUDA_VISIBLE_DEVICES=0,1,2

ray start --head \
    --num-cpus=48 \
    --num-gpus=3 \
    --port=6379 \
    --dashboard-port=8265 \
    --block \                  # --block keeps it alive; & backgrounds it
    # --node-ip-address=$(hostname -i)  & # important on clusters

RAY_HEAD_PID=$!
sleep 15                        # wait for Ray head to fully initialize
echo "Ray cluster started."

# ── veRL GRPO training ──────────────────────────────────────────────────────

# data.filter_overlong_prompts=True \
# +actor_rollout_ref.rollout.enforce_eager=True \

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    \
    data.train_files=src/output_RL/train.parquet \
    data.val_files=src/output_RL/test.parquet \
    data.train_batch_size=18 \
    data.max_prompt_length=6000 \
    data.max_response_length=2048 \
    data.truncation=left \
    \
    actor_rollout_ref.model.path=Qwen/Qwen3-8B \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    +actor_rollout_ref.model.torch_dtype=bfloat16 \
    +actor_rollout_ref.model.attention_implementation=sdpa \
    actor_rollout_ref.model.lora_rank=16 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.target_modules=all-linear \
    \
    actor_rollout_ref.actor.optim.lr=2e-5 \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.actor.ppo_mini_batch_size=18 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.dtype=bfloat16 \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=6 \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.top_k=20 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.max_model_len=8192 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    +actor_rollout_ref.rollout.disable_custom_all_reduce=True \
    \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.fsdp_config.dtype=bfloat16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    \
    reward_model.enable=False \
    custom_reward_function.path=src/training/verl_/reward.py \
    custom_reward_function.name=compute_score \
    \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name=causal_alignment \
    trainer.experiment_name=qwen3_8b_grpo \
    trainer.n_gpus_per_node=3 \
    trainer.nnodes=1 \
    trainer.save_freq=500 \
    trainer.test_freq=100 \
    trainer.total_epochs=3 \
    trainer.default_local_dir=src/output_RL/verl_checkpoints

TRAIN_EXIT=$?

# ── Cleanup ─────────────────────────────────────────────────────────────────
ray stop
kill "$JUDGE_PID" 2>/dev/null
wait "$RAY_HEAD_PID" 2>/dev/null

echo "Training exit code: $TRAIN_EXIT"
exit $TRAIN_EXIT