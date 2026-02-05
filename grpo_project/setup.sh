#!/bin/bash
# setup.sh

# 1. Create Directory (Already handled if running this script from inside grpo_project, but for completeness)
# mkdir -p grpo_project/src
# cd grpo_project

# 2. Create Conda Environment
conda create --name grpo_env python=3.10 -y
source $(conda info --base)/etc/profile.d/conda.sh
conda activate grpo_env

# 3. Install Pytorch (CUDA 12.1)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 4. Install Unsloth (Core optimization)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# 5. Install Dependencies (TRL, Accelerate, etc.)
pip install --no-deps "trl<0.9.0" peft accelerate bitsandbytes
pip install datasets scipy tensorboard flash-attn protobuf
