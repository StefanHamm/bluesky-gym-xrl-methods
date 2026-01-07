#!/bin/bash
#SBATCH --job-name=blueskyWrapper
#SBATCH --output=logs/slurm/slurm_%A_%a.out
#SBATCH --error=logs/slurm/slurm_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-4
#SBATCH --partition=sunnycove
# #SBATCH --gres=gpu:1
# #SBATCH --partition=gpu

mkdir -p logs/${SLURM_ARRAY_JOB_ID}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate blueskyGym

TOTAL_TIMESTEPS=2000000

ENV_IDX=1
ALGO_IDX=$((SLURM_ARRAY_TASK_ID % 5))
WRAPPER_IDX=0


echo "Running Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Environment Index: $ENV_IDX"
echo "Algorithm Index: $ALGO_IDX"

JOBDIR="logs/${SLURM_ARRAY_JOB_ID}"
mkdir -p $JOBDIR


#making workdir for each job
WORKDIR="workdirs/job_${SLURM_ARRAY_TASK_ID}_env${ENV_IDX}_algo${ALGO_IDX}"
mkdir -p $WORKDIR

# Run the python script with the specific indices
python src/train_baselines/train_wrapper_envs.py --env_idx $ENV_IDX --algo_idx $ALGO_IDX --num_cpu $SLURM_CPUS_PER_TASK --total_timesteps $TOTAL_TIMESTEPS --workdir $WORKDIR --jobdir $JOBDIR --jobid $SLURM_ARRAY_JOB_ID --wrapper_idx $WRAPPER_IDX