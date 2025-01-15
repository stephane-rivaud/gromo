#!/bin/bash

# Constants for growth parameters
hidden_size_list=(1)
epoch_per_growth_list=(1)
weight_decay_list=(0.0)

# Function to create the slurm directory
setup_environment() {
  mkdir -p slurm
}

# Function to execute the batch jobs
run_jobs() {
  local num_blocks=1
  local selection_method='none'

  for weight_decay in "${weight_decay_list[@]}"; do
    for hidden_size in "${hidden_size_list[@]}"; do
      for epoch_per_growth in "${epoch_per_growth_list[@]}"; do
        local command="scripts/mlp_run.sh $num_blocks $hidden_size $weight_decay $epoch_per_growth $selection_method"
        echo $command
        sbatch --gres=gpu:1 --time=01:45:00 $command
      done
    done
  done
}

# Main execution
setup_environment
run_jobs
