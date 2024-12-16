#!/bin/bash

# Source the shell configuration file to apply changes (need for conda)
source /home/tau/strivaud/.bashrc

# Create the conda environment
conda create -n gromo python=3.12

# Activate the conda environment
conda activate gromo

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -e .
pip install mlflow

# Test the PyTorch installation and the GPU
python -c "import torch; print(torch.cuda.is_available())"