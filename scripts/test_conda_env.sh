#!/bin/bash

#SBATCH --job-name=test-env
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err

# Source the shell configuration file to apply changes (need for conda)
source /home/tau/strivaud/.bashrc

# activate the conda environment
conda activate gromo

# Test the PyTorch installation and the GPU
python misc/test_conda_env.py | tee logs/test_conda_env.log