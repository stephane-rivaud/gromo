#!/bin/bash

# Constants for growth parameters
hidden_size_list=(2 4 8 16 32 64)
epoch_per_growth_list=(-1 0 1 2 4 8 16)

# Function to create the slurm directory
setup_environment() {
  mkdir -p slurm
}

# Function to execute the batch jobs
run_jobs() {
  local nb_hidden_layer=1
  local weight_decay=0
  local selection_method='none'

  for hidden_size in "${hidden_size_list[@]}"; do
    for epoch_per_growth in "${epoch_per_growth_list[@]}"; do
      local command="scripts/mlp_run.sh $nb_hidden_layer $hidden_size $weight_decay $epoch_per_growth $selection_method"
      echo $command
      sbatch --gres=gpu:1 --time=01:45:00 $command
    done
  done
}

# Main execution
setup_environment
run_jobs
