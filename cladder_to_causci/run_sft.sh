#!/bin/bash
#SBATCH --account=a0133
#SBATCH --job-name=cladder_sft
#SBATCH --output=/iopsstor/scratch/cscs/ajannali/project/sft_runs/cladder_sft_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/ajannali/project/sft_runs/cladder_sft_%j.err
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --partition=normal
#SBATCH --time=06:00:00
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#
# CLadder six-step SFT (LoRA, DDP). Phase A (turn-by-turn) → Phase B1 (single-pass, warm-started
# from A). The single-pass merged model (output/sft/final) is what RL loads as its base.
# One-time: mkdir -p /iopsstor/scratch/cscs/ajannali/project/sft_runs
set -e

# venv on scratch (created once on the LOGIN node under this uenv, --system-site-packages to
# inherit torch/transformers from the image, + pip install peft networkx pandas pyarrow).
# Compute nodes have no internet, so do NOT pip install here — deps are pre-provisioned.
source /iopsstor/scratch/cscs/ajannali/venv/cladder/bin/activate

PROJECT=/iopsstor/scratch/cscs/ajannali/project/causal_alignment
DIR=$PROJECT/cladder_to_causci
cd "$DIR"

# ── config (override any of these from the environment: `SFT_BASE=... sbatch run_sft.sh`) ──
export SFT_BASE=${SFT_BASE:-Qwen/Qwen3-8B}                          # base model (train_sft reads this)
export HF_HOME=${HF_HOME:-/iopsstor/scratch/cscs/ajannali/hf_cache} # HF cache (pre-download here on login node)
GPUS=${GPUS:-4}                                                     # GPUs / torchrun procs
BS=${BS:-4}                                                         # per-device batch size
ACCUM=${ACCUM:-4}                                                   # gradient accumulation steps
LR=${LR:-1e-4}                                                      # learning rate
EPOCHS_A=${EPOCHS_A:-1}                                             # Phase A (turn-by-turn) epochs
EPOCHS_B=${EPOCHS_B:-2}                                             # Phase B1 (single-pass) epochs
export SFT_EVAL_EVERY=${SFT_EVAL_EVERY:-100}                        # held-out eval every N steps
export SFT_EVAL_N=${SFT_EVAL_N:-256}                               # rows per periodic eval
echo "SFT cfg: base=$SFT_BASE gpus=$GPUS bs=$BS accum=$ACCUM lr=$LR epochsA=$EPOCHS_A epochsB=$EPOCHS_B eval_every=$SFT_EVAL_EVERY eval_n=$SFT_EVAL_N"

# NOTE: writes to output/sft/. Archive a completed run yourself before re-running, e.g.
#   mv output/sft/{final,turnsA,grade.json,metrics.csv,phase_metrics.jsonl} output/sft/v2/
rm -f output/sft/metrics.csv output/sft/grade.json output/sft/phase_metrics.jsonl   # fresh eval logs

# ── data: all-rung rlvr jsonl + parquet + SFT jsonl (turns + single-pass) ────
python3 data.py

# ── baseline eval: BASE model before any training (phase-snapshot for the plot) ──
python3 train_sft.py --grade "$SFT_BASE" --phase base || true

# NOTE: use `python -m torch.distributed.run` (NOT `torchrun`). Under the uenv, `torchrun` is the
# image's script and runs the IMAGE python (no peft); the module form runs the activated venv python.

# ── Phase A: turn-by-turn (learn each step; watch graph_f1 → ~1.0) ──────────
python -m torch.distributed.run --nproc_per_node=$GPUS train_sft.py \
    --train output/sft_turns.jsonl --out output/sft/turnsA \
    --epochs $EPOCHS_A --bs $BS --accum $ACCUM --lr $LR
python3 train_sft.py --grade output/sft/turnsA --phase A || true    # post-Phase-A snapshot

# ── Phase B1: single-pass, warm-started from Phase A (collapse the scaffold) ──
python -m torch.distributed.run --nproc_per_node=$GPUS train_sft.py \
    --train output/sft_single.jsonl --init output/sft/turnsA --out output/sft/final \
    --epochs $EPOCHS_B --bs $BS --accum $ACCUM --lr $LR

# ── post-SFT diagnostic + SFT→RL handoff gate (graph extraction must be near-solved) ──
# Non-fatal here (writes grade.json + phase B snapshot) so the job always finishes; the gate is
# ENFORCED by run_rl.sh, which refuses to start if graph_f1 < 0.95.
python3 train_sft.py --grade output/sft/final --phase B || echo "[handoff] graph_f1 below gate — RL will refuse until improved"

echo "SFT done → output/sft/final  (run_rl.sh warm-starts from it)"
# CauSci transfer is now measured DURING RL (run_rl.sh adds causci_val.parquet as a 2nd val set);
# the SFT-stage baseline = the first [causci_eval] pass in RL (before/at step 0, warm-started from SFT).
