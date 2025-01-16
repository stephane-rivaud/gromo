#!/bin/bash

# Constants for growth parameters
#num_block_list=(1 2 4 6)
#num_features=(256 512 1024)
#hidden_size_list=(32 64 128)
#epoch_per_growth_list=(-1 8 4)
#weight_decay_list=(0.0 0.01)

num_block_list=(10)
num_features=(1024)
hidden_size_list=(256)
epoch_per_growth_list=(-1)
weight_decay_list=(0.0 0.01)

# Function to create the slurm directory
setup_environment() {
  mkdir -p slurm
}

# Function to execute the batch jobs

run_jobs() {
  for weight_decay in "${weight_decay_list[@]}"; do
    for hidden_size in "${hidden_size_list[@]}"; do
      for num_features in "${num_features[@]}"; do
        for num_blocks in "${num_block_list[@]}"; do
          for epoch_per_growth in "${epoch_per_growth_list[@]}"; do
            if [ "$num_blocks" == 1 ]; then
              selection_method='none'
            else
              selection_method='fo'
            fi
            local command="scripts/residual_mlp_run.sh $num_blocks $num_features $hidden_size $weight_decay $epoch_per_growth $selection_method"
            echo $command
            sbatch --gres=gpu:1 --time=01:45:00 $command
          done
        done
      done
    done
  done
}

# Main execution
setup_environment
run_jobs
