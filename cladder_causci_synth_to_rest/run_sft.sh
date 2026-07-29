#!/bin/bash
#SBATCH --account=a0133
#SBATCH --job-name=clad_causci_sft
#SBATCH --output=/iopsstor/scratch/cscs/ajannali/project/sft_runs/clad_causci_sft_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/ajannali/project/sft_runs/clad_causci_sft_%j.err
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --partition=normal
#SBATCH --time=10:00:00
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#
# CauSci synth→{real,qr} SFT, but warm-started from the CLadder-SFT model instead of base Qwen3-8B.
# Same code as causci_synth_to_rest (reuses its train_sft.py); only the base model + data/out dirs change.
# Measures whether CLadder reasoning-SFT helps or hurts CauSci transfer. Needs the CLadder run finished
# (its output/sft/final must exist).
set -e

source /iopsstor/scratch/cscs/ajannali/venv/cladder/bin/activate

PROJECT=/iopsstor/scratch/cscs/ajannali/project/causal_alignment
CAUSCI=$PROJECT/causci_synth_to_rest          # reuse its train_sft.py + generated data
CLAD=$PROJECT/cladder_to_causci
NEW=$PROJECT/cladder_causci_synth_to_rest

# ── the only real change: base model = CLadder-SFT output (a full merged HF dir) ──
export SFT_BASE=${SFT_BASE:-$CLAD/output/sft/final}   # warm-start from CLadder-SFT model
export SFT_DATA_DIR=${SFT_DATA_DIR:-$CAUSCI/data}     # same synth train + real/qr eval as causci_synth_to_rest
export SFT_OUT_DIR=${SFT_OUT_DIR:-$NEW/output/phaseB}        # write here, not in causci_synth_to_rest
export HF_HOME=${HF_HOME:-/iopsstor/scratch/cscs/ajannali/hf_cache}

GPUS=${GPUS:-4}; BS=${BS:-4}; ACCUM=${ACCUM:-4}; LR=${LR:-1e-4}; EPOCHS=${EPOCHS:-2}
export SFT_EVAL_EVERY=${SFT_EVAL_EVERY:-25}; export SFT_EVAL_N=${SFT_EVAL_N:-0}
echo "clad→causci SFT: base=$SFT_BASE gpus=$GPUS bs=$BS accum=$ACCUM lr=$LR epochs=$EPOCHS"

[ -d "$SFT_BASE" ] || { echo "MISSING base model $SFT_BASE — finish the CLadder SFT first"; exit 1; }
for f in "$SFT_DATA_DIR/train.jsonl" "$SFT_DATA_DIR/eval.jsonl"; do
    [ -s "$f" ] || { echo "MISSING $f — run causci_synth_to_rest/build_data.py first"; exit 1; }
done

mkdir -p "$SFT_OUT_DIR/sft"
rm -f "$SFT_OUT_DIR/sft"/{metrics.csv,grade.json,phase_metrics.jsonl}
cd "$CAUSCI"                                          # train_sft.py lives here; DATA/OUT come from env

# ── baseline: CLadder-SFT model's CauSci transfer BEFORE any CauSci training ──
python3 train_sft.py --grade "$SFT_BASE" --phase base || true

# NOTE: `python -m torch.distributed.run` (NOT torchrun) so it uses the venv python (has peft).
python -m torch.distributed.run --nproc_per_node=$GPUS train_sft.py \
    --train "$SFT_DATA_DIR/train.jsonl" --out "$SFT_OUT_DIR/sft/final" \
    --epochs $EPOCHS --bs $BS --accum $ACCUM --lr $LR

python3 train_sft.py --grade "$SFT_OUT_DIR/sft/final" --phase final || true

echo "clad→causci SFT done → $SFT_OUT_DIR/sft/final"
