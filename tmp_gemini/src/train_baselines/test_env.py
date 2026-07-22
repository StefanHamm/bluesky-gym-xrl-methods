
import gymnasium as gym
from stable_baselines3 import SAC,PPO,TD3,DDPG,A2C
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import argparse
import sys
import bluesky_gym
import bluesky as bs
import bluesky_gym.envs
import os
import collections
from bluesky_gym.utils import logger
import numpy as np
import random
import torch


bluesky_gym.register_envs()
#env_name = "FreeFlightCREnv-v0"
#env_name = "PlanWaypointEnv-v2"
env_name = "PlanWaypointEvadeEnv-v0"
#all_envs = ["FreeFlightCREnv-v0"]
algorithms = [SAC, PPO, TD3, DDPG, A2C]

env=gym.make(env_name,render_mode="human",training = True) #training true for the training version of the env, false for the testing version, which has more information in the observation space and a different reward function.  
env.metadata["render_fps"] = 100  # Set the desired FPS for rendering

model = SAC("MultiInputPolicy", env, verbose=0, learning_rate=3e-4, seed=41)
#model = SAC.load(f"models/01/{env_name}/checkpoints/{env_name}_SAC_vecEnvLogs_baseline_2000000_steps.zip")
#model = SAC.load(r"models\01\FreeFlightCREnv-v0\checkpoints\FreeFlightCREnv-v0_SAC_vecEnvLogs_baseline_500000_steps.zip")    
model = SAC.load(r"models\4901832\PlanWaypointEvadeEnv-v0\PlanWaypointEvadeEnv-v0_SAC_vecEnvLogs_baseline_model_mp.zip")
env.reset(seed=41)

for x in range(19):
    obs,info = env.reset()
    
    total_reaward = 0
    
    for x in range(500):
        action, _states = model.predict(obs, deterministic=True)
        #ction = np.array([0.])
        obs, reward, terminated, truncated, info = env.step(action)
        total_reaward += reward
        
        #print(f"Step {x}: Action: {action}, Reward: {reward}, Total Reward: {total_reaward}")
        agent_idx = bs.traf.id2idx('KL001')
        #print(bs.traf.hdg[agent_idx])
        #print(f"Should be {bs.traf.hdg[agent_idx] *-0.5}")
       
        if terminated or truncated:
            break
    print(f"Episode reward: {total_reaward}")

env.close()