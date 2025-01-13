#!/bin/bash

# Constants for growth parameters
hidden_size_list=(512)
epoch_per_growth_list=(-1)

# Function to create the slurm directory
setup_environment() {
  mkdir -p slurm
}

# Function to execute the batch jobs
run_jobs() {
  local nb_hidden_layer=5
  local weight_decay=0.0
  local selection_method='none'

  for hidden_size in "${hidden_size_list[@]}"; do
    for epoch_per_growth in "${epoch_per_growth_list[@]}"; do
      local command="scripts/mlp_run.sh $nb_hidden_layer $hidden_size $weight_decay $epoch_per_growth $selection_method"
      echo $command
      sbatch --partition tau --gres=gpu:1 --time=01:45:00 $command
    done
  done
}

# Main execution
setup_environment
run_jobs
