#!/bin/bash
#SBATCH --job-name=train_static_v1
#SBATCH --output=logs/slurm/train_static_v1_%j.out
#SBATCH --error=logs/slurm/train_static_v1_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=sunnycove
# #SBATCH --gres=gpu:1
# #SBATCH --partition=gpu

source ~/miniconda3/etc/profile.d/conda.sh
conda activate blueskyGym

TOTAL_TIMESTEPS=2000000
ENV_IDX=4
ALGO_IDX=0

echo "Running Training Job: $SLURM_JOB_ID"
echo "Environment Index: $ENV_IDX (StaticObstacleEnv-v1)"
echo "Algorithm Index: $ALGO_IDX (SAC)"

# making workdir for the job
WORKDIR="workdirs/job_${SLURM_JOB_ID}_env${ENV_IDX}_algo${ALGO_IDX}"
mkdir -p $WORKDIR

# Run the python script with the specific indices
~/miniconda3/envs/blueskyGym/bin/python src/train_baselines/train_single_baseline_env.py \
    --env_idx $ENV_IDX \
    --algo_idx $ALGO_IDX \
    --num_cpu $SLURM_CPUS_PER_TASK \
    --total_timesteps $TOTAL_TIMESTEPS \
    --make_vec_env \
    --workdir $WORKDIR \
    --jobid $SLURM_JOB_ID
