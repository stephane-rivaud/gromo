#!/bin/bash

hidden_size=1
weight_decay=0

for nb_hidden_layer in '1'; do
  for epoch_per_growth in '2'; do
    for selection_method in 'none'; do
      command="scripts/mlp_run_test.sh $nb_hidden_layer $hidden_size $weight_decay $epoch_per_growth $selection_method"
      # DEBUG: Print command to verify it
      echo "Generated Command: $command"
      eval "ls -l scripts/mlp_run_test.sh"
      bash $command # Execute the command
    done
  done
done
