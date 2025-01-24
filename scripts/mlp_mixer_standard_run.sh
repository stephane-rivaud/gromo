#!/bin/bash

#SBATCH --job-name=mlp-mixer
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err

# Source shell configuration files
[ -f /home/tau/strivaud/.bashrc ] && source /home/tau/strivaud/.bashrc
[ -f /home/strivaud/.bash_profile ] && source /home/tau/strivaud/.bash_profile

# Activate conda environment
conda activate gromo

# Command to execute
command="python -u misc/mlp_mixer_standard_run.py"

# General arguments
num_workers=4
command+=" --num-workers $num_workers"

# Dataset arguments
dataset="cifar10"
nb_class=10
dataset_path="dataset"
data_augmentation="randaugment"

command+=" --dataset $dataset --nb-class $nb_class --dataset-path $dataset_path --data-augmentation $data_augmentation"

# Model arguments
num_blocks=8
num_features=128
hidden_dim_token=64
hidden_dim_channel=512
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
# Set the GPU to use
export CUDA_VISIBLE_DEVICES=0

# Execute the command
echo "Executing command: $command"
eval "$command"