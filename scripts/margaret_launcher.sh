#!/bin/bash

# Constants for growth parameters
no_cuda=false
sbatch scripts/mlp_mixer_run.sh $no_cuda

no_cuda=true
sbatch scripts/mlp_mixer_run.sh $no_cuda
