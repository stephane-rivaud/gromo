#!/bin/bash

# Source the shell configuration file to apply changes (need for conda)
source /home/tau/strivaud/.bashrc

# Delete the conda environment if it already exists
conda env remove -n gromo -y

# Create the conda environment
conda create -n gromo python=3.12 -y

# Activate the conda environment
conda activate gromo

# Install dependencies
#pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu118
pip install -e .
pip install mlflow tqdm

# Test the PyTorch installation and the GPU
sbatch scripts/test_conda_env.sh