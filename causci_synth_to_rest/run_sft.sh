#!/bin/bash
#SBATCH --account=a0133
#SBATCH --job-name=synth2rest_sft
#SBATCH --output=/iopsstor/scratch/cscs/ajannali/project/sft_runs/synth2rest_sft_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/ajannali/project/sft_runs/synth2rest_sft_%j.err
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --partition=normal
#SBATCH --time=10:00:00
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#
# CauSci synth→{real,qr} transfer SFT (LoRA, DDP, single-pass). Trains on data/train.jsonl (synth),
# evaluates transfer on data/eval.jsonl (real+qr) periodically + at base/final. Merged model → output/sft/final.
# Data processing is separate — this assumes data/{train,eval}.jsonl already exist.
# One-time: mkdir -p /iopsstor/scratch/cscs/ajannali/project/sft_runs
set -e

# same venv as cladder_to_causci (torch/transformers/peft/pandas inherited from the uenv image;
# compute nodes have no internet, so deps are pre-provisioned — do NOT pip install here).
source /iopsstor/scratch/cscs/ajannali/venv/cladder/bin/activate

PROJECT=/iopsstor/scratch/cscs/ajannali/project/causal_alignment
DIR=$PROJECT/causci_synth_to_rest
cd "$DIR"

# ── config (override from the environment: `SFT_BASE=... EPOCHS=3 sbatch run_sft.sh`) ──
export SFT_BASE=${SFT_BASE:-Qwen/Qwen3-8B}                          # base model (train_sft reads this)
export HF_HOME=${HF_HOME:-/iopsstor/scratch/cscs/ajannali/hf_cache} # HF cache (pre-download on login node)
GPUS=${GPUS:-4}
BS=${BS:-4}
ACCUM=${ACCUM:-4}
LR=${LR:-1e-4}
EPOCHS=${EPOCHS:-2}
export SFT_EVAL_EVERY=${SFT_EVAL_EVERY:-100}                        # transfer eval every N steps
export SFT_EVAL_N=${SFT_EVAL_N:-0}                                  # cap eval rows (0 → all real+qr)
echo "synth2rest SFT: base=$SFT_BASE gpus=$GPUS bs=$BS accum=$ACCUM lr=$LR epochs=$EPOCHS eval_every=$SFT_EVAL_EVERY"

# data-processing produces these (separate step); fail early if missing
for f in data/train.jsonl data/eval.jsonl; do
    [ -s "$f" ] || { echo "MISSING $DIR/$f — run your data-processing step first"; exit 1; }
done

# writes to output/sft/. Archive a finished run yourself before re-running:
#   mv output/sft/{final,grade.json,metrics.csv,phase_metrics.jsonl} output/sft/v1/
rm -f output/sft/metrics.csv output/sft/grade.json output/sft/phase_metrics.jsonl

# ── baseline: BASE model transfer before any training (snapshot for the plot) ──
python3 train_sft.py --grade "$SFT_BASE" --phase base || true

# NOTE: use `python -m torch.distributed.run` (NOT `torchrun`). Under the uenv, `torchrun` is the
# image's script and runs the IMAGE python (no peft); the module form runs the activated venv python.
python -m torch.distributed.run --nproc_per_node=$GPUS train_sft.py \
    --train data/train.jsonl --out output/sft/final \
    --epochs $EPOCHS --bs $BS --accum $ACCUM --lr $LR

# ── final transfer snapshot (real+qr) ──
python3 train_sft.py --grade output/sft/final --phase final || true

echo "synth2rest SFT done → output/sft/final"
