#!/bin/bash
#SBATCH --account=a0107
#SBATCH --job-name=cai_sft_training
#SBTACH --output=result.out
#SBATCH --error=results.err
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#SBATCH --time=06:00:00

# uenv image pull pytorch/v2.9.1:v2
uenv start pytorch/v2.9.1:v2 --view=default

source /iopsstor/scratch/cscs/ajannali/venv/cai/bin/activate
cd /iopsstor/scratch/cscs/ajannali/project/causal_alignment

python -m torch.distributed.run --nproc_per_node=4 ./src/training/train_sft_ddp.py
# python $SCRATCH/project/causal_alignment/src/training/train.py
# python $SCRATCH/project/causal_alignment/src/training/train_sft.py
