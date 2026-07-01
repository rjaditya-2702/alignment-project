#!/bin/bash
#SBATCH --account=a0133
#SBATCH --job-name=cai_sft
#SBATCH --output=/iopsstor/scratch/cscs/ajannali/project/sft_runs/sft_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/ajannali/project/sft_runs/sft_%j.err
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --partition=normal
#SBATCH --time=06:00:00
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#
# SFT phase (LoRA, DDP). One-time setup before first run:
#   mkdir -p /iopsstor/scratch/cscs/ajannali/project/sft_runs
set -e

# uenv start pytorch/v2.9.1:v2 --view=default
source /iopsstor/scratch/cscs/ajannali/venv/cai_trl/bin/activate

PROJECT=/iopsstor/scratch/cscs/ajannali/project/causal_alignment
cd "$PROJECT/new_code"

# eval generates on the test set and scores via library_fn (statsmodels/linearmodels/sklearn)
pip install peft scikit-learn statsmodels linearmodels scipy rdd

# 1. Build train/test jsonl: qr+synth → train_sft / train_rl (35/65), real → test_{sft,rl}.
#    Single process (not under torchrun) so it runs once.
python3 data_prep.py

# 2. SFT: trains on train_sft.jsonl, evals on test_sft.jsonl every SFT_EVAL_EVERY steps
#    (writes output/sft_metrics.csv) and saves the merged model to output/sft/final.
