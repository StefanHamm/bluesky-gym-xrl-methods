#!/bin/bash
#SBATCH --job-name=blueskyBaselines
#SBATCH --output=logs/slurm/slurm_%A_%a.out
#SBATCH --error=logs/slurm/slurm_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=18 
#SBATCH --mem=32G
#SBATCH --array=0-4
#SBATCH --partition=broadwell
# #SBATCH --gres=gpu:1
# #SBATCH --partition=gpu

mkdir -p logs/${SLURM_ARRAY_JOB_ID}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate blueskyGym

TOTAL_TIMESTEPS=6000000

ALGO_IDX=$SLURM_ARRAY_TASK_ID


echo "Running Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Algorithm Index: $ALGO_IDX"


#making workdir for each job
WORKDIR="workdirs/job_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_algo${ALGO_IDX}"
mkdir -p $WORKDIR

# Run the python script with the specific indices
python src/train_baselines/train_nav_waypoint_evade_env.py --algo_idx $ALGO_IDX --num_cpu $SLURM_CPUS_PER_TASK --total_timesteps $TOTAL_TIMESTEPS --workdir $WORKDIR --jobid $SLURM_ARRAY_JOB_ID --make_vec_env

#cleanup workdir after training
rm -rf $WORKDIR