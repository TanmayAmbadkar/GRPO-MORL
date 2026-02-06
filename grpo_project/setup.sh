#!/bin/bash
# setup.sh

# 1. Create and Activate Environment
# Using 3.10 as it's the most stable for Unsloth/Torch dependencies
conda create --name grpo_env python=3.10 -y
source $(conda info --base)/etc/profile.d/conda.sh
conda activate grpo_env

# 2. Install Torch (CUDA 12.1)
# Using pip ensures we get the exact wheel we need without conda channel conflicts.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Install Unsloth & Core Dependencies
# We removed flash-attn to avoid the build errors.
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "trl<0.9.0" peft accelerate bitsandbytes
pip install datasets scipy tensorboard protobuf rewardbench

echo "Setup Complete without flash-attn. Unsloth will use PyTorch SDPA instead."
echo "Activate with: 'conda activate grpo_env'"
