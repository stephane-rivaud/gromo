#!/bin/bash

# Constants for growth parameters
for bs in 128 512 2048 8192; do
    sbatch scripts/mlp_mixer_run.sh $bs
done
