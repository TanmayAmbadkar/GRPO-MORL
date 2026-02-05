# D3PO GRPO Project

This repository supports the D3PO (Direct Preference Optimization with Group Relative Policy Optimization) system for a 4x GPU setup.

## Project Structure

### Core Implementation
- **[main.py](main.py)**: The main orchestrator script that manages the training loop, multi-GPU model loading (Policy, Reference, Reward), and optimization.
- **[src/utils.py](src/utils.py)**: Contains the math helper functions including `get_batch_logps` for proper masking, `sample_weights` for Dirichlet preference sampling, and `compute_grouped_advantages` for GRPO normalization.
- **[src/reward_engine.py](src/reward_engine.py)**: The `MultiObjectiveRewardEngine` class that wraps the ArmoRM model to provide scores for Helpfulness, Truthfulness, Safety, and Content Quality.
- **[src/loss.py](src/loss.py)**: The custom `D3POGRPOLoss` function that computes the weighted PPO loss across multiple objectives.

### Setup
- **[setup.sh](setup.sh)**: A shell script containing the commands to create the conda environment and install all dependencies.

## Usage Instructions

1.  **Transfer Files**: If you are not running this on the Mac where these files were generated, copy the entire `grpo_project` folder to your Linux cluster.
2.  **Environment Setup**:
    ```bash
    cd grpo_project
    ./setup.sh
    ```
    *Note: If you are setting this up manually, you can open `setup.sh` to see the individual commands.*
3.  **Run Training**:
    ```bash
    conda activate grpo_env
    python main.py --batch_size 1 --group_size 4 --ppo_epochs 2 --beta_kl 0.05 --max_steps 500
    ```

## Hardware Configuration
- **GPU 0**: Policy Model (Trainable LoRA)
- **GPU 1**: Reference Model (Frozen)
- **GPU 2**: Reward Model (ArmoRM)
- **GPU 3**: Spare
