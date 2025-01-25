#!/bin/bash

# Constants for growth parameters
for bs in 128 512 2048 8192; do
    sbatch run_training.sh $bs $epochs $optimizer $scheduler
done
