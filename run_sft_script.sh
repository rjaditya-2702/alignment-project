#!/bin/bash
#SBATCH --account=a0107
#SBATCH --job-name=cai_sft_training
#SBATCH --output=/iopsstor/scratch/cscs/ajannali/project/sft_runs/result_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/ajannali/project/sft_runs/results_%j.err
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --partition=normal
#SBATCH --time=06:00:00
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default

# uenv image pull pytorch/v2.9.1:v2
uenv start pytorch/v2.9.1:v2 --view=default

source /iopsstor/scratch/cscs/ajannali/venv/cai_trl/bin/activate
cd /iopsstor/scratch/cscs/ajannali/project/causal_alignment

python -m torch.distributed.run --nproc_per_node=4 ./src/training/train_sft_ddp.py
# python $SCRATCH/project/causal_alignment/src/training/train.py
# python $SCRATCH/project/causal_alignment/src/training/train_sft.py
