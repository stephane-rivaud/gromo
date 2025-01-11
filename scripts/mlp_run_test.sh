#!/bin/bash

# SLURM parameters
#SBATCH --job-name=mlp-run
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err

# Extract script parameters
nb_hidden_layer=$1
hidden_size=$2
weight_decay=$3
epochs_per_growth=$4
selection_method=$5

# Debugging: Print received arguments
if [ -z "$nb_hidden_layer" ]; then
  echo "Error: nb_hidden_layer is not set"
  exit 1
fi
if [ -z "$hidden_size" ]; then
  echo "Error: hidden_size is not set"
  exit 1
fi
if [ -z "$weight_decay" ]; then
  echo "Error: weight_decay is not set"
  exit 1
fi
if [ -z "$epochs_per_growth" ]; then
  echo "Error: epochs_per_growth is not set"
  exit 1
fi
if [ -z "$selection_method" ]; then
  echo "Error: selection_method is not set"
  exit 1
fi

# Print debug information
echo "nb_hidden_layer: $nb_hidden_layer"
echo "hidden_size: $hidden_size"
echo "weight_decay: $weight_decay"
echo "epochs_per_growth: $epochs_per_growth"
echo "selection_method: $selection_method"

command="python -u misc/mlp_run.py"

# Append required arguments
command="${command} --nb-hidden-layer $nb_hidden_layer"
command="${command} --hidden-size $hidden_size"
command="${command} --weight-decay $weight_decay"
command="${command} --epochs-per-growth $epochs_per_growth"
command="${command} --selection-method $selection_method"

# Optional debugging
echo "Final command to execute: $command"

# Execute command
eval "$command"