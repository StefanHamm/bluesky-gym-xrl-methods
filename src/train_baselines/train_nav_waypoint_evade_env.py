"""
This file trains a basline model using a single environment and a specified algorithm.
"""

import gymnasium as gym
from stable_baselines3 import SAC,PPO,TD3,DDPG,A2C
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv,VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
import argparse
import sys
import bluesky_gym
import bluesky_gym.envs
import os
import collections
from bluesky_gym.utils import logger
import numpy as np
import random
import torch

def set_global_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


bluesky_gym.register_envs()

keywords_mapping = {
    "NavWaypointEvadeEnv-v0": [
        "drift_mean",
        "corridor_leave_mean",
        "intrusion_count",
        "obstacle_intrusion_count",
        "waypoint_reached_count",
        "path_length",
        "crash"
    ],
    # Add more mappings for other environments as needed
}


ENV_NAME = "NavWaypointEvadeEnv-v0"
ALGORITHMS = [SAC, PPO, TD3, DDPG, A2C]

def make_env():

    if args.workdir:
        env = gym.make(ENV_NAME, render_mode=None, workdir=args.workdir)
        sys.stdout.flush()
        return env
    else:
        env = gym.make(ENV_NAME, render_mode=None)
        return env



TRAIN = True




if __name__ == "__main__":
    global env_name
    # 1. Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo_idx", type=int, default=0, help="Index of algorithm in algorithms list")
    parser.add_argument("--num_cpu", type=int, default=2, help="Number of CPUs to use")
    parser.add_argument("--total_timesteps", type=float, default=1e2, help="Total training timesteps")
    parser.add_argument("--workdir", type=str, default=None, help="Working directory for BlueSky sim")
    parser.add_argument("--make_vec_env", action='store_true', help="Use vectorized environment")
    parser.add_argument("--jobid", type=str, default=None, help="Job identifier")
    parser.add_argument("--save_best_model", action='store_true', help="Save best model during training")
    parser.add_argument("--model_seed", type=int, default=42, help="Random seed for model")
    parser.add_argument("--env_seed", type=int, default=0, help="Random seed for environment")
    parser.add_argument("--global_seed",type=int,default=42 , help="Set global random seed, for randomness that may be outside model/env")
    args = parser.parse_args()


    #output parsed arguments for logging
    print(f"Algorithm Index: {args.algo_idx}")
    print(f"Number of CPUs: {args.num_cpu}")
    print(f"Total Timesteps: {args.total_timesteps}")
    print(f"Working Directory: {args.workdir}")
    print(f"Use Vectorized Environment: {args.make_vec_env}")
    print(f"Job ID: {args.jobid}")
    print(f"Save Best Model: {args.save_best_model}")
    print(f"Model Seed: {args.model_seed}")
    print(f"Environment Seed: {args.env_seed}")

    set_global_seed(args.global_seed)
    # 2. Select specific config
    try:
        algorithm = ALGORITHMS[args.algo_idx]
    except IndexError:
        print("Algorithm index out of bounds")
        sys.exit(1)

    print(f"--- Starting Job: {algorithm.__name__} on {ENV_NAME} ---")
    print(f"{args.total_timesteps}")

    if args.make_vec_env:
        suffix = "vecEnvLogs"
    else:
        suffix = "singleEnv"

    log_dir = f'./logs/{args.jobid}/{ENV_NAME}/'
    file_name = f'{ENV_NAME}_{str(algorithm.__name__)}_{suffix}_baseline'
    log_file_path = os.path.join(log_dir, file_name)
    # create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    if TRAIN:
        if args.make_vec_env:
            print("Using vectorized environment")
            env = make_vec_env(make_env,seed=args.env_seed, n_envs=args.num_cpu, vec_env_cls=SubprocVecEnv)
            env = VecMonitor(env, filename=log_file_path, info_keywords=keywords_mapping.get(ENV_NAME, []))
        else:
            print("Using single environment")
            env = make_env(logger_path=log_file_path)
            env = Monitor(env, log_file_path, info_keywords=keywords_mapping.get(ENV_NAME, []))
            env.reset(seed=args.env_seed)

        policy_type = "MultiInputPolicy"
        model = algorithm(policy_type, env, verbose=0, learning_rate=3e-4, seed=args.model_seed,device="cuda" if torch.cuda.is_available() else "cpu")
        model.learn(total_timesteps=int(args.total_timesteps),  progress_bar=False)
        model.save(f"models/{args.jobid}/{ENV_NAME}/{ENV_NAME}_{str(algorithm.__name__)}_{suffix}_baseline_model_mp")
        env.close()