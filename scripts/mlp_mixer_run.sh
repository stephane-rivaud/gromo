#!/bin/bash

#SBATCH --job-name=mlp-mixer
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err

# Script parameters


echo "TMPDIR: $TMPDIR"
echo "num_blocks: $num_blocks"
echo "epochs_per_growth: $epochs_per_growth"
echo "selection_method: $selection_method"

# Source shell configuration files
[ -f /home/tau/strivaud/.bashrc ] && source /home/tau/strivaud/.bashrc
[ -f /home/strivaud/.bash_profile ] && source /home/tau/strivaud/.bash_profile

# Activate conda environment
conda activate gromo

# Command to execute
command="python -u misc/mlp_mixer_run.py"

# General arguments
log_dir="logs/mlp_mixer_run"
mkdir -p $log_dir
experiment_name="MLP_mixer-${num_blocks}_blocks"
tags="mlp-mixer"
log_system_metrics=true
num_workers=4

command+=" --log-dir $log_dir --experiment-name $experiment_name --tags $tags --num-workers $num_workers"
[ "$log_system_metrics" = true ] && command+=" --log-system-metrics"

# Dataset arguments
dataset="cifar10"
nb_class=10
split_train_val=0.0
dataset_path="dataset"
data_augmentation="randaugment"

command+=" --dataset $dataset --nb-class $nb_class --split-train-val $split_train_val --dataset-path $dataset_path --data-augmentation $data_augmentation"

# Model arguments
num_blocks=8
num_features=128
hidden_dim_token=2
hidden_dim_channel=16
bias=true

command+=" --num-blocks $num_blocks --num-features $num_features --hidden-dim-token $hidden_dim_token --hidden-dim-channel $hidden_dim_channel"
[ "$bias" = false ] && command+=" --no-bias"

# Training arguments
nb_step=300
batch_size=128
optimizer="adamw"
lr=1e-3
dropout=0.0
weight_decay=5e-5

command+=" --nb-step $nb_step --batch-size $batch_size --optimizer $optimizer --lr $lr --weight-decay $weight_decay --dropout $dropout"

# Scheduler arguments
scheduler="cosine"
warmup_epochs=5

command+=" --scheduler $scheduler --warmup-epochs $warmup_epochs"

# Growing training arguments
epochs_per_growth=$1
growing_batch_limit=-1
growing_part="all"
growing_numerical_threshold=1e-5
growing_statistical_threshold=1e-3
growing_maximum_added_neurons=10
growing_computation_dtype="float32"
selection_method='fo'
normalize_weights=false
init_new_neurons_with_random_in_and_zero_out=false

command+=" --epochs-per-growth $epochs_per_growth --selection-method $selection_method --growing-batch-limit $growing_batch_limit --growing-part $growing_part --growing-numerical-threshold $growing_numerical_threshold --growing-statistical-threshold $growing_statistical_threshold --growing-maximum-added-neurons $growing_maximum_added_neurons --growing-computation-dtype $growing_computation_dtype"
[ "$normalize_weights" = true ] && command+=" --normalize-weights"
[ "$init_new_neurons_with_random_in_and_zero_out" = true ] && command+=" --init-new-neurons-with-random-in-and-zero-out"

# Line search arguments
line_search_alpha=0.1
line_search_beta=0.5
line_search_max_iter=20
line_search_epsilon=1e-7
line_search_batch_limit=-1

command+=" --line-search-alpha $line_search_alpha --line-search-beta $line_search_beta --line-search-max-iter $line_search_max_iter --line-search-epsilon $line_search_epsilon --line-search-batch-limit $line_search_batch_limit"

# Set the GPU to use
export CUDA_VISIBLE_DEVICES=0

# Execute the command
echo "Executing command: $command"
eval "$command"