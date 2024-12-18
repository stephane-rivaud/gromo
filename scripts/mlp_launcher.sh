#!/bin/bash

mkdir -p slurm

# model arguments
nb_hidden_layer=4
hidden_size=10

# optimization arguments
weight_decay=0

# growth arguments
epoch_per_growth=10
selection_method="none" # "none", "fo"

for nb_hidden_layer in 1 2; do
  for hidden_size in 8 16 32; do
    for weight_decay in 0 0.0001 0.001; do
      for epoch_per_growth in -1 1 2; do
        for selection_method in "none" "fo"; do
          command="scripts/mlp_run.sh $nb_hidden_layer $hidden_size $weight_decay $epoch_per_growth $selection_method"
          echo $command
          sbatch --gres=gpu:1 --time=04:00:00 --partition titanic $command
        done
      done
    done
  done
done
