#!/bin/bash
#SBATCH --account=a0133
#SBATCH --job-name=causal_verl_grpo
#SBATCH --output=/iopsstor/scratch/cscs/ajannali/project/verl_runs/verl_train_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/ajannali/project/verl_runs/verl_train_%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --time=10:00:00


cd /iopsstor/scratch/cscs/ajannali/project/causal_alignment

RESUME=true
rm -rf /iopsstor/scratch/cscs/ajannali/project/judge_server.log
rm -rf /iopsstor/scratch/cscs/ajannali/project/verl_runs
rm -rf /iopsstor/scratch/cscs/ajannali/project/causal_alignment/core_nid*

rm -rf /iopsstor/scratch/cscs/ajannali/project/causal_alignment/judge_server.log
rm -rf /iopsstor/scratch/cscs/ajannali/project/causal_alignment/verl_runs
rm -rf /iopsstor/scratch/cscs/ajannali/project/core_nid*

if [ "$RESUME" = "false" ]; then
    rm -rf /iopsstor/scratch/cscs/ajannali/project/causal_alignment/verl_training_test.log
    rm -rf /iopsstor/scratch/cscs/ajannali/project/causal_alignment/verl_metrics_test.csv
    rm -rf /iopsstor/scratch/cscs/ajannali/project/causal_alignment/src/output_RL/verl_checkpoints/*
fi

# set -x
EDF=$SCRATCH/project/env_toml.toml
VENV=$SCRATCH/verl
PROJECT=/iopsstor/scratch/cscs/ajannali/project/causal_alignment
PORT=6379

# ── resolve nodes & IPs (no ssh) ───────────────────────────────────────────
nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
head_node=${nodes[0]}
worker_node=${nodes[1]}
export HEAD_IP=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | awk '{print $1}')
WORKER_IP=$(srun --nodes=1 --ntasks=1 -w "$worker_node" hostname --ip-address | awk '{print $1}')
echo "HEAD=$head_node ($HEAD_IP)  WORKER=$worker_node ($WORKER_IP)"

# ── Data preparation (runs once, rank-0 only via fcntl in the script) ──────
srun --overlap --nodes=1 --ntasks=1 -w "$head_node" --environment="$EDF" bash -c '
  source '"$VENV"'/bin/activate
  cd '"$PROJECT"'
  echo "Running data preparation..."
  python3 src/training/verl_/data_process.py
  if [ $? -ne 0 ]; then
    echo "ERROR: data_process.py failed." >&2
    exit 1
  fi
  echo "Data preparation complete."
'

SIZES=$(srun --overlap --nodes=1 --ntasks=1 -w "$head_node" --environment="$EDF" bash -c '
  source '"$VENV"'/bin/activate
  cd '"$PROJECT"'
  python3 -c "import math, pandas as pd; df = pd.read_parquet(\"src/output_RL/train.parquet\"); n = len(df); steps=3*math.ceil(n/40); print(n, steps, max(15, steps//100))"
')

TRAIN_SIZE_N=$(echo "$SIZES" | awk '{print $1}')
TOTAL_STEPS=$(echo "$SIZES"  | awk '{print $2}')
TEST_FREQ=$(echo "$SIZES"    | awk '{print $3}')
echo "train_size=$TRAIN_SIZE_N  total_steps=$TOTAL_STEPS  TEST_FREQ=$TEST_FREQ"

CKPT_DIR=/iopsstor/scratch/cscs/ajannali/project/causal_alignment/src/output_RL/verl_checkpoints
LATEST_CKPT=$(find "$CKPT_DIR" -maxdepth 1 -type d -name "global_step_*" | sort -V | tail -n 1)
 
RESUME_ARG=""
RESUME=true
if [ "${RESUME:-}" = "true" ] && [ -n "$LATEST_CKPT" ]; then
    RESUME_ARG="trainer.resume_mode=resume_path trainer.resume_from_path=$LATEST_CKPT"
    echo "Resuming from: $LATEST_CKPT"
fi

# judge URL that reward.py must call — reachable from BOTH nodes
export JUDGE_URL="http://$HEAD_IP:8001/v1"

# ── node 1: the trainer (foreground; connects to the running Ray cluster) ──
srun --overlap -N1 -n1 -w "$head_node" --environment="$EDF" bash -c '
    source '"$VENV"'/bin/activate; cd '"$PROJECT"'

    CUDA_VISIBLE_DEVICES=3 python3 -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen3-8B --port 8001 --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.18 --max-model-len 1500 --max-num-seqs 4 \
        --dtype bfloat16 > '"$PROJECT"'/judge_server.log 2>&1 &
    JPID=$!

    for i in $(seq 1 120); do
        curl -sf http://localhost:8001/health >/dev/null 2>&1 && { echo "Judge up"; break; }
        kill -0 $JPID 2>/dev/null || { echo "ERROR: judge died"; tail -20 '"$PROJECT"'/judge_server.log; exit 1; }
        [ "$i" = "120" ] && { echo "ERROR: judge timeout"; exit 1; }
        sleep 5
    done

    unset CUDA_VISIBLE_DEVICES
    unset ROCR_VISIBLE_DEVICES
    unset HIP_VISIBLE_DEVICES



    ray start --head --node-ip-address='"$HEAD_IP"' --port=6379 --num-gpus=4 --num-cpus=48
    sleep 8
    for i in $(seq 1 60); do [ $(ray status 2>/dev/null | grep -c node_) -ge 2 ] && break; sleep 5; done
    ray status

    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export PYTHONPATH='"$PROJECT"':$PYTHONPATH
    export JUDGE_URL=http://'"$HEAD_IP"':8001/v1

    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        algorithm.use_kl_in_reward=False \
        \
        data.train_batch_size=20 \
        data.max_prompt_length=2250 data.max_response_length=2250 \
        data.truncation=left data.dataloader_num_workers=4 data.shuffle=True \
        +data.apply_chat_template_kwargs.enable_thinking=true \
        +data.apply_chat_template_kwargs.thinking_budget=650 \
        data.train_files='"$PROJECT"'/src/output_RL/train.parquet \
        data.val_files='"$PROJECT"'/src/output_RL/test.parquet \
        \
        actor_rollout_ref.model.path=Qwen/Qwen3-8B \
        actor_rollout_ref.model.lora_rank=16 actor_rollout_ref.model.lora_alpha=32 \
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
        actor_rollout_ref.actor.ppo_mini_batch_size=20 \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=5000 \
        actor_rollout_ref.actor.loss_agg_mode=token-mean \
        \
        actor_rollout_ref.rollout.n=6 actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.top_k=10 actor_rollout_ref.rollout.top_p=0.95 \
        actor_rollout_ref.rollout.temperature=0.7 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.free_cache_engine=True \
        actor_rollout_ref.rollout.dtype=bfloat16 \
        actor_rollout_ref.rollout.max_model_len=4500 \
        actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
        actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=9000 \
        actor_rollout_ref.rollout.enforce_eager=False \
        \
        actor_rollout_ref.ref.fsdp_config.param_offload=False \
        actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
        actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=9000 \
        \
        reward_model.enable=False \
        custom_reward_function.path='"$PROJECT"'/src/training/verl_/reward.py \
        custom_reward_function.name=compute_score \
        \
        trainer.val_before_train=False \
        trainer.critic_warmup=0 \
        trainer.logger=[console,file] \
        trainer.project_name=causal_alignment trainer.experiment_name=qwen3_8b_grpo \
        trainer.n_gpus_per_node=4 trainer.nnodes=2 \
        trainer.save_freq=50 trainer.test_freq=20 \
        trainer.total_epochs=3 \
        trainer.default_local_dir='"$PROJECT"'/src/output_RL/verl_checkpoints \
        2>&1 | tee -a '"$PROJECT"'/verl_training_test.log
' &
A_PID=$!

srun --overlap -N1 -n1 -w "$worker_node" --environment="$EDF" bash -c "

    unset CUDA_VISIBLE_DEVICES
    unset ROCR_VISIBLE_DEVICES
    unset HIP_VISIBLE_DEVICES
    
    source $VENV/bin/activate; cd $PROJECT
    until ray start --address=$HEAD_IP:6379 --num-gpus=4 --num-cpus=48 --block; do
        echo 'head not ready, retrying in 5s'; sleep 5
    done
" &
B_PID=$!

# A_PID = head node: judge + ray head + trainer
# B_PID = worker node: ray worker (blocks on --block)

# Wait specifically for the trainer. -n returns when the FIRST job finishes,
# but we want to key off A_PID specifically, so wait on it by PID:
wait "$A_PID"
A_RC=$?

# Trainer has exited (success or failure). The worker's `ray start --block`
# will never return on its own, so kill it regardless.
echo "Trainer (A_PID=$A_PID) exited with code $A_RC; tearing down worker."
kill "$B_PID" 2>/dev/null
wait "$B_PID" 2>/dev/null

exit "$A_RC"