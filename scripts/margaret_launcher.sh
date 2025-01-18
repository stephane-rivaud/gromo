#!/bin/bash

# Constants for growth parameters
#num_block_list=(1 2 4 6)
#num_features=(256 512 1024)
#hidden_size_list=(32 64 128)
#epoch_per_growth_list=(-1 8 4)
#weight_decay_list=(0.0 0.0005)

num_block_list=(4)
num_features=(128)
hidden_size_list=(32)
epoch_per_growth_list=(-1)
weight_decay_list=(0.0 0.0005)
dropout_list=(0.0 0.3)

# Function to create the slurm directory
setup_environment() {
  mkdir -p slurm
}

# Function to execute the batch jobs

run_jobs() {
  for hidden_size in "${hidden_size_list[@]}"; do
    for num_features in "${num_features[@]}"; do
      for num_blocks in "${num_block_list[@]}"; do
        for epoch_per_growth in "${epoch_per_growth_list[@]}"; do
          for weight_decay in "${weight_decay_list[@]}"; do
            for dropout in "${dropout_list[@]}"; do
              if [ "$num_blocks" == 1 ]; then
                selection_method='none'
              else
                selection_method='fo'
              fi
              local command="scripts/mlp_mixer_run.sh $num_blocks $num_features $hidden_size $weight_decay $epoch_per_growth $selection_method $dropout"
              echo $command
              sbatch --partition tau --gres=gpu:1 --time=01:45:00 $command
            done
          done
        done
      done
    done
  done
}

# Main execution
setup_environment
run_jobs
