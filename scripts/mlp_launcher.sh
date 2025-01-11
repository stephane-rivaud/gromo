#!/bin/bash

mkdir -p slurm

# model arguments
nb_hidden_layer=1
hidden_size=64

# optimization arguments
weight_decay=0

# growth arguments
for nb_hidden_layer in 2; do
  for epoch_per_growth in -1; do
    for selection_method in 'none'; do
      command="scripts/mlp_run.sh $nb_hidden_layer $hidden_size $weight_decay $epoch_per_growth $selection_method"
      echo $command
      sbatch --gres=gpu:1 --time=01:45:00 $command
    done
  done
done
