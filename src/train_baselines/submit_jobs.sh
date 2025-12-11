#!/bin/bash
#SBATCH --job-name=bluesky_baselines
#SBATCH --output=logs/slurm/slurm_%A_%a.out
#SBATCH --error=logs/slurm/slurm_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --array=0-19



source ~/miniconda3/etc/profile.d/conda.sh
conda activate blueskyGym

TOTAL_TIMESTEPS=2048

ENV_IDX=$((SLURM_ARRAY_TASK_ID / 5))
ALGO_IDX=$((SLURM_ARRAY_TASK_ID % 5))


echo "Running Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Environment Index: $ENV_IDX"
echo "Algorithm Index: $ALGO_IDX"

# Run the python script with the specific indices
python src/train_baselines/train_single_baseline_env.py --env_idx $ENV_IDX --algo_idx $ALGO_IDX --num_cpu $SLURM_CPUS_PER_TASK --total_timesteps $TOTAL_TIMESTEPS