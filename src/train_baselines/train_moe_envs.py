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
    "FreeFlightCREnv-v0": ['total_intrusions'],
    "PlanWaypointEnv-v2": ['waypoints_completed'],
    "PlanWaypointEvadeEnv-v0": ['waypoints_completed','total_intrusions']
    # Add more mappings for other environments as needed
}


all_envs = ["FreeFlightCREnv-v0","PlanWaypointEnv-v2","PlanWaypointEvadeEnv-v0"]
algorithms = [SAC, PPO, TD3, DDPG, A2C]

def make_env(logger_path=None):
    """
    Utility function for multiprocessed env.
    """
    if args.workdir:
        os.makedirs(args.workdir, exist_ok=True)
    # ...existing code...
    if env_name == "SectorCREnv-v0":
        env = gym.make(env_name, render_mode=None)
    else:
        env = gym.make(env_name, render_mode=None, workdir=args.workdir)
    return env



TRAIN = True




if __name__ == "__main__":
    global env_name
    # 1. Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_idx", type=int, default=0, help="Index of environment in all_envs list")
    parser.add_argument("--algo_idx", type=int, default=0, help="Index of algorithm in algorithms list")
    parser.add_argument("--num_cpu", type=int, default=2, help="Number of CPUs to use")
    parser.add_argument("--total_timesteps", type=float, default=1e2, help="Total training timesteps")
    parser.add_argument("--workdir", type=str, default=None, help="Working directory for BlueSky sim")
    parser.add_argument("--make_vec_env", action='store_true', help="Use vectorized environment")
    parser.add_argument("--jobdir", type=str, default=None, help="Job directory for logs")
    parser.add_argument("--jobid", type=str, default=None, help="Job identifier")
    parser.add_argument("--save_best_model", action='store_true', help="Save best model during training")
    parser.add_argument("--model_seed", type=int, default=42, help="Random seed for model")
    parser.add_argument("--env_seed", type=int, default=0, help="Random seed for environment")
    parser.add_argument("--global_seed",type=int,default=42 , help="Set global random seed, for randomness that may be outside model/env")
    args = parser.parse_args()

    set_global_seed(args.global_seed)
    # 2. Select specific config
    try:
        env_name = all_envs[args.env_idx]
        algorithm = algorithms[args.algo_idx]
    except IndexError:
        print("Index out of bounds")
        sys.exit(1)

    print(f"--- Starting Job: {algorithm.__name__} on {env_name} ---")
    print(f"{args.total_timesteps}")

    # create a folder called vecEnvLogs and store the logs there if it is false store it in singleEnvLogs
    if args.make_vec_env:
        suffix = "vecEnvLogs"
    else:
        suffix = "singleEnv"

    # 3. Run Training (No loops here anymore!)
    log_dir = f'./{args.jobdir}/{env_name}/'
    file_name = f'{env_name}_{str(algorithm.__name__)}_{suffix}_baseline'
    log_file_path = os.path.join(log_dir, file_name)
    
    #csv_logger_callback = logger.CSVLoggerCallback(log_dir, file_name)
    
    if TRAIN:
        if args.make_vec_env:
            print("Using vectorized environment")
            env = make_vec_env(make_env,seed=args.env_seed, n_envs=args.num_cpu, vec_env_cls=SubprocVecEnv)
            env = VecMonitor(env, filename=log_file_path, info_keywords=keywords_mapping.get(env_name, []))
        else:
            print("Using single environment")
            env = make_env(logger_path=log_file_path)
            env = Monitor(env, log_file_path, info_keywords=keywords_mapping.get(env_name, []))
            env.reset(seed=args.env_seed)
        
    
        policy_type = "MultiInputPolicy"
        
        model = algorithm(policy_type, env, verbose=0, learning_rate=3e-4, seed=args.model_seed,device="cuda" if torch.cuda.is_available() else "cpu")
        
        # Calculate frequency roughly equivalent to 100k timesteps
        n_envs = args.num_cpu if args.make_vec_env else 1
        save_freq = max(100000 // n_envs, 1)
        
        # Save checkpoints
        checkpoint_dir = f"models/{args.jobid}/{env_name}/checkpoints"
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path=checkpoint_dir,
            name_prefix=f"{env_name}_{str(algorithm.__name__)}_{suffix}_baseline",
            verbose=1
        )
        
        model.learn(total_timesteps=int(args.total_timesteps),  progress_bar=True, callback=checkpoint_callback)
        model.save(f"models/{args.jobid}/{env_name}/{env_name}_{str(algorithm.__name__)}_{suffix}_baseline_model_mp")
        
        env.close()




