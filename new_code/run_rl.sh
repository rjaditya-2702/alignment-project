#!/bin/bash
#SBATCH --account=a0133
#SBATCH --job-name=cai_rl_grpo
#SBATCH --output=/iopsstor/scratch/cscs/ajannali/project/verl_runs/rl_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/ajannali/project/verl_runs/rl_%j.err
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --partition=normal
#SBATCH --time=06:00:00
#SBATCH --environment=/iopsstor/scratch/cscs/ajannali/project/env_toml.toml
#
# RL phase (veRL GRPO). Base model = the SFT checkpoint. No judge server: CauSci
# verification is programmatic (library_fn), so all 4 GPUs go to training.
# One-time setup before first run:
#   mkdir -p /iopsstor/scratch/cscs/ajannali/project/verl_runs

PROJECT=/iopsstor/scratch/cscs/ajannali/project/causal_alignment
NEW=$PROJECT/new_code
cd "$NEW"

RESUME=true
CKPT_DIR=$NEW/output/rl/verl_checkpoints
BASE=$NEW/output/sft/final          # ← RL continues from the SFT-trained model

if [ "$RESUME" = "false" ]; then
    rm -rf "$NEW/verl_training.log" "$NEW/output/rl_metrics.csv" "$CKPT_DIR"/*
fi

# ── deps (match the veRL env pins) ──────────────────────────────────────────
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

# ── data: build rl/test jsonl → parquet (no cladder, no judge) ──────────────
python3 data_prep.py
python3 rl_data.py

# The SFT checkpoint's tokenizer was saved by a different transformers version
# (extra_special_tokens format drift) → re-save the base Qwen3 tokenizer with THIS env's
# transformers so veRL can load it. LoRA never changes the vocab, so it's the same tokenizer.
python3 -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen3-8B', trust_remote_code=True).save_pretrained('$BASE')"

# ── Ray (all 4 GPUs for training) ───────────────────────────────────────────
unset ROCR_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0,1,2,3
export RAY_USAGE_STATS_ENABLED=0
ray start --head --num-cpus=48 --num-gpus=4 --port=6379 --dashboard-port=8265 --block &
sleep 15
echo "Ray started."

TEST_FREQ=1   # eval on the real test set every training step (dense SFT→RL timeline)
echo "test_freq=$TEST_FREQ (eval every step)"

LATEST_CKPT=$(find "$CKPT_DIR" -maxdepth 1 -type d -name "global_step_*" 2>/dev/null | sort -V | tail -n 1)
RESUME_ARG=""
if [ "$RESUME" = "true" ] && [ -n "$LATEST_CKPT" ]; then
    RESUME_ARG="trainer.resume_mode=resume_path trainer.resume_from_path=$LATEST_CKPT"
    echo "Resuming from: $LATEST_CKPT"
fi

# NOTE: SFT trained no-think (direct JSON); here RL rolls out WITH thinking to develop
# reasoning. reward.extract_json strips </think>, so scoring is unaffected. Set
# enable_thinking=false below if you want SFT/RL eval modes to match exactly.
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    \
    data.train_files=[output/train_rl.parquet] \
    data.val_files=[output/test.parquet] \
    data.train_batch_size=20 \
    data.max_prompt_length=2500 \
    data.max_response_length=3000 \
    data.truncation=left \
    data.shuffle=True \
    +data.apply_chat_template_kwargs.enable_thinking=false \
    +data.apply_chat_template_kwargs.thinking_budget=512 \
    \
    actor_rollout_ref.model.path=$BASE \
    actor_rollout_ref.model.lora_rank=16 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.target_modules=all-linear \
    ++actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    +actor_rollout_ref.model.override_config.torch_dtype=bfloat16 \
    +actor_rollout_ref.model.override_config.attn_implementation=flash_attention_3 \
    \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.actor.ppo_mini_batch_size=20 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.05 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.top_k=15 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.max_model_len=5500 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    \
    reward_model.enable=False \
    custom_reward_function.path=$NEW/reward.py \
    custom_reward_function.name=compute_score \
    \
    trainer.critic_warmup=0 \
    $RESUME_ARG \
    trainer.logger='["console","file"]' \
    trainer.project_name=causal_alignment \
    trainer.experiment_name=qwen3_8b_grpo_causci \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=$TEST_FREQ \
    trainer.total_epochs=3 \
    trainer.default_local_dir=$CKPT_DIR \
    2>&1 | tee -a verl_training.log

TRAIN_EXIT=${PIPESTATUS[0]}

# ── parse [verl_eval] lines → rl_metrics.csv (feeds plot.py) ────────────────
python3 parse_rl_log.py verl_training.log output/rl_metrics.csv

ray stop
echo "Training exit code: $TRAIN_EXIT"
exit $TRAIN_EXIT
