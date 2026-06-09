"""
This file trains a basline model using a single environment and a specified algorithm.
"""

import gymnasium as gym
from stable_baselines3 import SAC,PPO,TD3,DDPG,A2C
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
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
from stable_baselines3.common.monitor import Monitor

def set_global_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


bluesky_gym.register_envs()

keywords_mapping = {
    "PlanWaypointEnv-v0": ['waypoints_completed'],
    "HorizontalCREnv-v0": ['total_intrusions','average_drift'],
    "SectorCREnv-v0": ['total_intrusions','average_drift'],
    "StaticObstacleEnv-v0": ['crashed','average_drift','waypoint_reached'],
    "VerticalCREnv-v0": ['total_intrusions', 'final_altitude'],
    "DescentEnv-v0": ['final_altitude'],
    "MergeEnv-v0": ['faf_reach', 'average_drift', 'total_intrusions']
}


# all_envs = ["SectorCREnv-v0","HorizontalCREnv-v0","StaticObstacleEnv-v0","PlanWaypointEnv-v0"]
# all_envs = ["VerticalCREnv-v0"]
#all_envs = ["SectorCREnv-v0","HorizontalCREnv-v0","StaticObstacleEnv-v0","PlanWaypointEnv-v0","VerticalCREnv-v0"]
all_envs = ["DescentEnv-v0", "VerticalCREnv-v0", "StaticObstacleEnv-v0", "MergeEnv-v0"]
algorithms = [SAC, PPO, TD3, DDPG, A2C]
#algorithms = [SAC, TD3]

def make_env():
    """
    Utility function for multiprocessed env.
    """
    if args.workdir:
        os.makedirs(args.workdir, exist_ok=True)
    # ...existing code...
    if env_name in ["StaticObstacleEnv-v0", "VerticalCREnv-v0"]:
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


    # create a folder called vecEnvLogs and store the logs there if it is false store it in singleEnvLogs
    if args.make_vec_env:
        suffix = "vecEnvLogs"
    else:
        suffix = "singleEnv"

    # 3. Run Training (No loops here anymore!)
    log_dir = f'./logs/{args.jobid}/{env_name}/'
    file_name = f'{env_name}_{str(algorithm.__name__)}_{suffix}_baseline'
    log_file_path = os.path.join(log_dir, file_name)
    
    #csv_logger_callback = logger.CSVLoggerCallback(log_dir, file_name)
    
    if TRAIN:
        if args.make_vec_env:
            print("Using vectorized environment")
            env = make_vec_env(make_env,seed=args.env_seed, n_envs=args.num_cpu, vec_env_cls=SubprocVecEnv)
            env = VecMonitor(env, filename=log_file_path,info_keywords=keywords_mapping[env_name])
        else:
            print("Using single environment")
            env = make_env()
            env.reset(seed=args.env_seed)
            env = Monitor(env,filename=log_file_path, info_keywords=tuple(keywords_mapping[env_name]))
        
    
        policy_type = "MultiInputPolicy"
        
        model = algorithm(policy_type, env, verbose=1, learning_rate=3e-4,
            seed=args.model_seed,
            tensorboard_log=f"./logs/tensorboard/{env_name}/")
        
        try:
            model.learn(total_timesteps=int(args.total_timesteps), progress_bar=False)
        except KeyboardInterrupt:
            print("Training interrupted. Saving intermediate model...")
        finally:
            model.save(f"models/{args.jobid}/{env_name}/{env_name}_{str(algorithm.__name__)}_{suffix}_baseline_model_mp")
            env.close()




