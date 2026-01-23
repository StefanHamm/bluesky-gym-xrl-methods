import gymnasium as gym
from stable_baselines3 import SAC,TD3,DDPG,PPO
import bluesky_gym
import bluesky_gym.envs
import shap
import numpy as np
import matplotlib.pyplot as plt
import logging
import copy
from bluesky_gym.wrappers.xrlMethods.state.horizontal_cr_env_action_heatmap import ActionHeatmapWrapper

from bluesky_gym.utils import logger
bluesky_gym.register_envs()
import time


if __name__ == "__main__":
    JOBID = "4675598"
    SEED = 42
    DEBUG = False
    # Initialize the environment and logger
    env_name = 'HorizontalCREnv-v0'

    if DEBUG:
        gifFolder = f"./plots/{JOBID}/{env_name}/actionHeatmapDebug/"
    else:
        gifFolder = f"./plots/{JOBID}/{env_name}/actionHeatmap/"

    env = gym.make(env_name,render_mode='human')
    env.reset(seed=SEED)
    

    modelpath = f"models/{JOBID}/HorizontalCREnv-v0/HorizontalCREnv-v0_SAC_singleEnv_baseline_model_mp.zip"
    #model = PPO.load(modelpath, env=saliencyEnv,device='cpu')
    model = SAC.load(modelpath,device='cpu')
    
    actionHeatmap = ActionHeatmapWrapper(env, model=model,draw_action_heatmap=True, grid_size=10, grid_spacing_km=5,export_gifs_path=gifFolder,fps=5)
    

    episodes = 20
    for ep in range(episodes):
        # skip first 10
        
        done = truncated = False
        obs, info = actionHeatmap.reset()
        step = 0
        while not (done or truncated):
            step+=1
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = actionHeatmap.step(action)