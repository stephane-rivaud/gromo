#!/bin/bash

mkdir -p slurm

# model arguments
nb_hidden_layer=1
hidden_size=1

# optimization arguments
weight_decay=0
epoch_per_growth=1
selection_method='none'

# growth arguments
for hidden_size in 1 2 4 8 16 32 64; do
  for epoch_per_growth in -1 0 1 2 4 6; do
    command="scripts/mlp_run.sh $nb_hidden_layer $hidden_size $weight_decay $epoch_per_growth $selection_method"
    echo $command
#      sbatch --gres=gpu:1 --time=01:45:00 $command
    eval "$command"
  done
done
