"""
This file is an example train and test loop for the different environments that
uses multiprocessing through the use of vectorised environments.
Note that multiprocessing doesn't necessarily result in faster training. It is
highly dependent on the environment and algorithm combination. If the algorithm
is able to train over a batch of observations, multiprocessing should lead to
faster training.
Selecting different environments is done through setting the 'env_name' variable.
"""

import gymnasium as gym
from stable_baselines3 import SAC,PPO,TD3,DDPG,A2C
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import argparse
import sys
import bluesky_gym
import bluesky_gym.envs
import os

from bluesky_gym.utils import logger

bluesky_gym.register_envs()

#env_name = 'SectorCREnv-v0'

all_envs = ["SectorCREnv-v0","HorizontalCREnv-v0","StaticObstacleEnv-v0","PlanWaypointEnv-v0"]
algorithms = [SAC, PPO, TD3, DDPG, A2C]
num_cpu = 2



def make_env():
    """
    Utility function for multiprocessed env.
    """
    global env_counter
    if args.workdir:
        os.makedirs(args.workdir, exist_ok=True)
    # ...existing code...
    if env_name == "StaticObstacleEnv-v0":
        env = gym.make(env_name, render_mode=None)
    else:
        env = gym.make(env_name, render_mode=None, workdir=args.workdir)
# ...existing code...
    # Set a different seed for each created environment.
    env.reset(seed=env_counter)
    env_counter +=1 
    return env

# Initialize logger
# log_dir = f'./logs/{env_name}/'
# file_name = f'{env_name}_{str(algorithm.__name__)}.csv'
# csv_logger_callback = logger.CSVLoggerCallback(log_dir, file_name)

TRAIN = True
EVAL_EPISODES = 10
TOTAL_TIMESTEPS = 1e2
# Initialise the environment counter
env_counter = 0



if __name__ == "__main__":
    MODEL_SEED = 42
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
    args = parser.parse_args()

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
    log_dir = f'./{args.jobdir}/{env_name}/'
    file_name = f'{env_name}_{str(algorithm.__name__)}_{suffix}_baseline.csv'
    csv_logger_callback = logger.CSVLoggerCallback(log_dir, file_name)
    
    # Reset global counter for this specific process
    env_counter = 0 
    
    if TRAIN:
        if args.make_vec_env:
            print("Using vectorized environment")
            env = make_vec_env(make_env, n_envs=args.num_cpu, vec_env_cls=SubprocVecEnv)
        else:
            print("Using single environment")
            env = make_env()
        
        if algorithm == RecurrentPPO:
            policy_type = "MultiInputLstmPolicy"
        else:
            policy_type = "MultiInputPolicy"
        
        if policy_type in ["SAC","TD3","DDPG"]:
            buffer_size = int(max(1e6,args.total_timesteps))
            model = algorithm(policy_type, env, verbose=0, learning_rate=3e-4, buffer_size=buffer_size, seed=MODEL_SEED)
        else:
        
            model = algorithm(policy_type, env, verbose=0, learning_rate=3e-4, seed=MODEL_SEED)
        
        
        model.learn(total_timesteps=int(args.total_timesteps), callback=csv_logger_callback, progress_bar=False)
        model.save(f"models/{args.jobid}/{env_name}/{env_name}_{str(algorithm.__name__)}_{suffix}_baseline_model_mp")
        
        env.close()




