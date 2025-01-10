#!/bin/bash

mkdir -p slurm

# model arguments
nb_hidden_layer=1
hidden_size=64

# optimization arguments
weight_decay=0

# growth arguments
epoch_per_growth=-1
selection_method="none" # "none", "fo", "scaled_fo", "one_step_fo"

command="scripts/mlp_run.sh $nb_hidden_layer $hidden_size $weight_decay $epoch_per_growth $selection_method"
echo $command
sbatch --gres=gpu:1 --time=01:45:00 $command
