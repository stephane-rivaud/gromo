#!/bin/bash

cluster_name=$1
if [ -z "$cluster_name" ]; then
  echo "Usage: $0 <cluster_name>"
  exit 1
fi

# Rsync command with a smooth progress bar using 'pv'
rsync -azh --info=progress2 strivaud@$cluster_name.saclay.inria.fr:/home/tau/strivaud/gromo/logs/res_mlp_run /Users/strivaud/PycharmProjects/gromo/logs
