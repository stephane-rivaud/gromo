#!/bin/bash

#SBATCH --job-name=mlp-run
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err

# script parameters
nb_hidden_layer=$1
hidden_size=$2
weight_decay=$3
epochs_per_growth=$4
selection_method=$5

# Source the shell configuration file to apply changes (need for conda)
if [ -f /home/tau/strivaud/.bashrc ]; then
    source /home/tau/strivaud/.bashrc
fi
if [ -f /home/strivaud/.bash_profile ]; then
    source /home/tau/strivaud/.bash_profile
fi
# activate the conda environment
conda activate gromo

command="python -u misc/mlp_run.py"

# General arguments
log_dir="logs/mlp_run"
log_dir_suffix=""
log_file_name=""
log_file_prefix=""
tags="mlp-test"
nb_step=100
no_cuda=false
training_threshold=""
log_system_metrics=true

command="${command} --nb-step $nb_step"
if [ -n "$log_dir" ]; then
    command="${command} --log-dir $log_dir"
fi
if [ -n "$log_dir_suffix" ]; then
    command="${command} --log-dir-suffix $log_dir_suffix"
fi
if [ -n "$log_file_name" ]; then
    command="${command} --log-file-name $log_file_name"
fi
if [ -n "$log_file_prefix" ]; then
    command="${command} --log-file-prefix $log_file_prefix"
fi
if [ -n "$tags" ]; then
    command="${command} --tags $tags"
fi
if [ "$no_cuda" = true ]; then
    command="${command} --no-cuda"
fi
if [ -n "$training_threshold" ]; then
    command="${command} --training-threshold $training_threshold"
fi
if [ "$log_system_metrics" = true ]; then
    command="${command} --log-system-metrics"
fi

# Dataset arguments
dataset="cifar10"
nb_class=10
split_train_val=0.3
dataset_path="dataset"
data_augmentation="horizontal_flip crop"

command="${command} --dataset $dataset --nb-class $nb_class --split-train-val $split_train_val --dataset-path $dataset_path"
if [ -n "$data_augmentation" ]; then
    command="${command} --data-augmentation $data_augmentation"
fi

# Model arguments
#nb_hidden_layer=4
#hidden_size=10
activation="selu"
bias=true

command="${command} --nb-hidden-layer $nb_hidden_layer --hidden-size $hidden_size --activation $activation"
if [ "$bias" = false ]; then
    command="${command} --no-bias"
fi

# Classical training arguments
#seed=0
batch_size=64
optimizer="sgd"
lr=0.01
#weight_decay=0

command="${command} --batch-size $batch_size --optimizer $optimizer --lr $lr --weight-decay $weight_decay"

# Growing training arguments
#epochs_per_growth=4
#selection_method="none"
growing_batch_limit=-1
growing_part="all"
growing_numerical_threshold=1e-5
growing_statistical_threshold=1e-3
growing_maximum_added_neurons=10
growing_computation_dtype="float32"
normalize_weights=false
init_new_neurons_with_random_in_and_zero_out=false

command="${command} --epochs-per-growth $epochs_per_growth --selection-method $selection_method --growing-batch-limit $growing_batch_limit --growing-part $growing_part --growing-numerical-threshold $growing_numerical_threshold --growing-statistical-threshold $growing_statistical_threshold --growing-maximum-added-neurons $growing_maximum_added_neurons --growing-computation-dtype $growing_computation_dtype"
if [ "$normalize_weights" = true ]; then
    command="${command} --normalize-weights"
fi
if [ "$init_new_neurons_with_random_in_and_zero_out" = true ]; then
    command="${command} --init-new-neurons-with-random-in-and-zero-out"
fi

## Line search arguments
line_search_alpha=0.1
line_search_beta=0.5
line_search_max_iter=20
line_search_epsilon=1e-7
line_search_batch_limit=-1

command="${command} --line-search-alpha $line_search_alpha --line-search-beta $line_search_beta --line-search-max-iter $line_search_max_iter --line-search-epsilon $line_search_epsilon --line-search-batch-limit $line_search_batch_limit"

# Execute the command
echo "Executing command: $command"
eval "$command"