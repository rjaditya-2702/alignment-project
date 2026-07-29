#!/bin/bash
#SBATCH --account=a0133
#SBATCH --job-name=cladder_sft
#SBATCH --output=/iopsstor/scratch/cscs/ajannali/project/sft_runs/cladder_sft_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/ajannali/project/sft_runs/cladder_sft_%j.err
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --partition=normal
#SBATCH --time=10:00:00
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

# REASON=1 → train on the reasoning-augmented data (build_reasoning.py must have run first, on the
# login node — it needs internet for the LLM API). Bigger budgets since traces inflate length.
if [ "${REASON:-0}" = "1" ]; then
    TURNS_F=output/sft_turns_reason.jsonl; SINGLE_F=output/sft_single_reason.jsonl
    export SFT_MAX_SEQ_LEN=${SFT_MAX_SEQ_LEN:-4096}; export SFT_GEN_MAX_NEW=${SFT_GEN_MAX_NEW:-1024}
    for f in "$TURNS_F" "$SINGLE_F"; do
        [ -s "$f" ] || { echo "MISSING $f — run build_reasoning.py (login node) first"; exit 1; }
    done
else
    TURNS_F=output/sft_turns.jsonl; SINGLE_F=output/sft_single.jsonl
fi
echo "SFT cfg: base=$SFT_BASE gpus=$GPUS bs=$BS accum=$ACCUM lr=$LR epochsA=$EPOCHS_A epochsB=$EPOCHS_B reason=${REASON:-0} turns=$TURNS_F single=$SINGLE_F"

# PHASE=AB (default) runs both; PHASE=A only Phase A; PHASE=B resumes Phase B from an existing
# output/sft/turnsA (e.g. after a walltime kill — no need to redo the expensive Phase A).
PHASE=${PHASE:-AB}

# NOTE: writes to output/sft/. Archive a completed run yourself before re-running, e.g.
#   mv output/sft/{final,turnsA,grade.json,metrics.csv,phase_metrics.jsonl} output/sft/v2/
[[ "$PHASE" == *A* ]] && rm -f output/sft/metrics.csv output/sft/grade.json output/sft/phase_metrics.jsonl  # fresh logs (full run only)

# ── data: all-rung rlvr jsonl + parquet + SFT jsonl (turns + single-pass) ────
python3 data.py

if [[ "$PHASE" == *A* ]]; then
    # ── baseline eval: BASE model before any training (phase-snapshot for the plot) ──
    python3 train_sft.py --grade "$SFT_BASE" --phase base || true

    # NOTE: use `python -m torch.distributed.run` (NOT `torchrun`). Under the uenv, `torchrun` is the
    # image's script and runs the IMAGE python (no peft); the module form runs the activated venv python.

    # ── Phase A: turn-by-turn (learn each step; watch graph_f1 → ~1.0) ──────────
    python -m torch.distributed.run --nproc_per_node=$GPUS train_sft.py \
        --train $TURNS_F --out output/sft/turnsA \
        --epochs $EPOCHS_A --bs $BS --accum $ACCUM --lr $LR
    python3 train_sft.py --grade output/sft/turnsA --phase A || true    # post-Phase-A snapshot
fi

if [[ "$PHASE" == *B* ]]; then
    [ -d output/sft/turnsA ] || { echo "MISSING output/sft/turnsA — run PHASE=A (or AB) first"; exit 1; }
    # ── Phase B1: single-pass, warm-started from Phase A (collapse the scaffold) ──
    python -m torch.distributed.run --nproc_per_node=$GPUS train_sft.py \
        --train $SINGLE_F --init output/sft/turnsA --out output/sft/final \
        --epochs $EPOCHS_B --bs $BS --accum $ACCUM --lr $LR

    # ── post-SFT diagnostic + SFT→RL handoff gate (graph extraction must be near-solved) ──
    # Non-fatal here (writes grade.json + phase B snapshot) so the job always finishes; the gate is
    # ENFORCED by run_rl.sh, which refuses to start if graph_f1 < 0.95.
    python3 train_sft.py --grade output/sft/final --phase B || echo "[handoff] graph_f1 below gate — RL will refuse until improved"
fi

echo "SFT done → output/sft/final  (run_rl.sh warm-starts from it)"
# CauSci transfer is now measured DURING RL (run_rl.sh adds causci_val.parquet as a 2nd val set);
# the SFT-stage baseline = the first [causci_eval] pass in RL (before/at step 0, warm-started from SFT).
