#!/bin/bash
#SBATCH --job-name=blueskyBaselines
#SBATCH --output=logs/slurm/slurm_%A_%a.out
#SBATCH --error=logs/slurm/slurm_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=3
#SBATCH --mem=32G
#SBATCH --array=0-19
#SBATCH --partition=sunnycove
# #SBATCH --gres=gpu:1
# #SBATCH --partition=gpu

mkdir -p logs/${SLURM_ARRAY_JOB_ID}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate blueskyGym

TOTAL_TIMESTEPS=2000000

ENV_IDX=$((SLURM_ARRAY_TASK_ID / 5))
ALGO_IDX=$((SLURM_ARRAY_TASK_ID % 5))


echo "Running Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Environment Index: $ENV_IDX"
echo "Algorithm Index: $ALGO_IDX"

#making workdir for each job
WORKDIR="workdirs/job_${SLURM_ARRAY_TASK_ID}_env${ENV_IDX}_algo${ALGO_IDX}"
mkdir -p $WORKDIR

# Run the python script with the specific indices
python src/train_baselines/train_single_baseline_env.py --env_idx $ENV_IDX --algo_idx $ALGO_IDX --num_cpu $SLURM_CPUS_PER_TASK --total_timesteps $TOTAL_TIMESTEPS --workdir $WORKDIR --jobid $SLURM_ARRAY_JOB_ID