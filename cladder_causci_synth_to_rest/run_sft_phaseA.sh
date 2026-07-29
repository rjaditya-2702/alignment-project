#!/bin/bash
#SBATCH --account=a0133
#SBATCH --job-name=clad_causciA_sft
#SBATCH --output=/iopsstor/scratch/cscs/ajannali/project/sft_runs/clad_causciA_sft_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/ajannali/project/sft_runs/clad_causciA_sft_%j.err
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --partition=normal
#SBATCH --time=10:00:00
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#
# CauSci synth→{real,qr} SFT warm-started from the CLadder Phase-A model (turnsA, turn-by-turn), NOT
# Phase B (final). Companion to run_sft.sh (which uses Phase B). Writes to output_phaseA/ so the two
# variants don't collide. Lets us compare which CLadder phase transfers better to CauSci.
set -e

source /iopsstor/scratch/cscs/ajannali/venv/cladder/bin/activate

PROJECT=/iopsstor/scratch/cscs/ajannali/project/causal_alignment
CAUSCI=$PROJECT/causci_synth_to_rest          # reuse its train_sft.py + generated data
CLAD=$PROJECT/cladder_to_causci
NEW=$PROJECT/cladder_causci_synth_to_rest

# ── base model = CLadder Phase A (turnsA), and a SEPARATE output dir ──
export SFT_BASE=${SFT_BASE:-$CLAD/output/sft/turnsA}   # Phase A (turn-by-turn) checkpoint
export SFT_DATA_DIR=${SFT_DATA_DIR:-$CAUSCI/data}
export SFT_OUT_DIR=${SFT_OUT_DIR:-$NEW/output/phaseA}  # distinct from run_sft.sh's output/
export HF_HOME=${HF_HOME:-/iopsstor/scratch/cscs/ajannali/hf_cache}

GPUS=${GPUS:-4}; BS=${BS:-4}; ACCUM=${ACCUM:-4}; LR=${LR:-1e-4}; EPOCHS=${EPOCHS:-2}
export SFT_EVAL_EVERY=${SFT_EVAL_EVERY:-25}; export SFT_EVAL_N=${SFT_EVAL_N:-0}
echo "cladA→causci SFT: base=$SFT_BASE gpus=$GPUS bs=$BS accum=$ACCUM lr=$LR epochs=$EPOCHS"

[ -d "$SFT_BASE" ] || { echo "MISSING base model $SFT_BASE — Phase A (turnsA) must exist"; exit 1; }
for f in "$SFT_DATA_DIR/train.jsonl" "$SFT_DATA_DIR/eval.jsonl"; do
    [ -s "$f" ] || { echo "MISSING $f — run causci_synth_to_rest/build_data.py first"; exit 1; }
done

mkdir -p "$SFT_OUT_DIR/sft"
rm -f "$SFT_OUT_DIR/sft"/{metrics.csv,grade.json,phase_metrics.jsonl}
cd "$CAUSCI"                                            # train_sft.py lives here; DATA/OUT come from env

# ── baseline: CLadder Phase-A model's CauSci transfer BEFORE any CauSci training ──
python3 train_sft.py --grade "$SFT_BASE" --phase base || true

python -m torch.distributed.run --nproc_per_node=$GPUS train_sft.py \
    --train "$SFT_DATA_DIR/train.jsonl" --out "$SFT_OUT_DIR/sft/final" \
    --epochs $EPOCHS --bs $BS --accum $ACCUM --lr $LR

python3 train_sft.py --grade "$SFT_OUT_DIR/sft/final" --phase final || true

echo "cladA→causci SFT done → $SFT_OUT_DIR/sft/final"
