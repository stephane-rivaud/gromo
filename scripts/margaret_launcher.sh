#!/bin/bash

# Constants for growth parameters
#num_block_list=(1 2 4 6)
#num_features=(256 512 1024)
#hidden_size_list=(32 64 128)
#epoch_per_growth_list=(-1 8 4)
#weight_decay_list=(0.0 0.0005)

epoch_per_growth_list=(-1 4 16 24 32)

# Function to create the slurm directory
setup_environment() {
  mkdir -p slurm
}

# Function to execute the batch jobs

run_jobs() {
  for epoch_per_growth in "${epoch_per_growth_list[@]}"; do
    local command="scripts/mlp_mixer_run.sh $epoch_per_growth"
    echo $command
    sbatch --partition tau --gres=gpu:1 --time=12:00:00 $command
  done
}

# Main execution
setup_environment
run_jobs
