#!/bin/bash

mkdir -p slurm

# model arguments
nb_hidden_layer=4
hidden_size=10

# optimization arguments
weight_decay=0

# growth arguments
epoch_per_growth=10
selection_method="none" # "none", "fo", "scaled_fo", "one_step_fo"

for nb_hidden_layer in 1 2 3; do
  for hidden_size in 8 16 32 64; do
    for weight_decay in 0 0.0001 0.001 0.01; do

      epoch_per_growth=-1
      selection_method="none"
      command="scripts/mlp_run.sh $nb_hidden_layer $hidden_size $weight_decay $epoch_per_growth $selection_method"
      echo $command
      sbatch --gres=gpu:1 --time=01:45:00 $command

      for epoch_per_growth in 0 1 2 4 8; do
        for selection_method in "none" "fo"; do
          command="scripts/mlp_run.sh $nb_hidden_layer $hidden_size $weight_decay $epoch_per_growth $selection_method"
          echo $command
          sbatch --gres=gpu:1 --time=01:45:00 $command
        done
      done

    done
  done
done
