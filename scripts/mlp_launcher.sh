#!/bin/bash

# model arguments
nb_hidden_layer=4
hidden_size=10

# optimization arguments
weight_decay=0

# growth arguments
epoch_per_growth=10
selection_method="none" # "none", "fo", "scaled_fo", "one_step_fo"

for nb_hidden_layer in 1 2; do
  for hidden_size in 8 16 32 64; do
    for weight_decay in 0 0.0001 0.001 0.01; do
      for epoch_per_growth in -1 1 2 4 8; do
        for selection_method in "none" "fo" "scaled_fo" "one_step_fo"; do
          command="mlp_run.sh --nb-hidden-layer $nb_hidden_layer --hidden-size $hidden_size --weight-decay $weight_decay --epochs-per-growth $epoch_per_growth --selection-method $selection_method"
          echo $command
          sbtach --gres=gpu:1 --time=00:45:00 $command
        done
      done
    done
  done
done
