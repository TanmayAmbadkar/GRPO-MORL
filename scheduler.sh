#!/bin/bash
#PBS -l ngpus=4
#PBS -l ncpus=32
#PBS -l walltime=24:00:00
#PBS -q workq@e5-cse-cbgpu01.eecscl.psu.edu
#PBS -N grpo_4gpu
#PBS -M tsa5252@psu.edu
#PBS -m bea
#PBS -l mem=100g

# Change to the directory where the job was submitted from
cd $PBS_O_WORKDIR

# Initialize conda properly for non-interactive shell
source /scratch/tsa5252/anaconda3/etc/profile.d/conda.sh

# Activate the environment
conda activate grpo_env

# Set HuggingFace cache to local directory (models pre-downloaded there)
export HF_HOME=/scratch/tsa5252/.cache/huggingface

# Navigate to project directory and run
cd /scratch/tsa5252/GRPO-MORL/grpo_project
accelerate launch --num_processes 4 main.py