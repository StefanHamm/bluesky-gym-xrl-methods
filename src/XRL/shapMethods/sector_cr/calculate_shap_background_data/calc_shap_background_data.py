"""Example script how to calculate and visualize SHAP values for background data explanations in HorizontalCR environment."""

import gymnasium as gym
from stable_baselines3 import SAC,TD3,DDPG,PPO
import bluesky_gym
import numpy as np
import logging
from bluesky_gym.wrappers.xrlMethods.state.saliency.sector_cr_env_saliency import SaliencySectorControl
from src.XRL.shapMethods.shap_explainers import runBackgroundExplainer
import numpy as np


from bluesky_gym.utils import logger
import os
bluesky_gym.register_envs()

# Initialize the environment and logger
env_name = 'SectorCREnv-v0'

if __name__ == "__main__":
    intruder_feature_mapping = ["x_r","y_r","vx_r","vy_r",
        "cos(track)","sin(track)","distances"]
    JOBID = "4675598"
    SEED = 42
    DEBUG = False
    RUN_BASELINE_ACTION = False

    color_mode = "clipped"  #"clipped"  #"scaled"
    if DEBUG:
        gifFolder = f"./plots/{JOBID}/{env_name}/shapBackgroundDataDebug/"
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    else:
        gifFolder = f"./plots/{JOBID}/{env_name}/shapBackgroundData/{color_mode}/"
        logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
        
    env = gym.make(env_name, render_mode="human")
    
    # Spawns more intruders than original env
    spawnFactor = 2
    options = {"SpawnFactor":spawnFactor}
    
    env.reset(seed=SEED,options=options)
    saliencyEnv = SaliencySectorControl(env,None,None,export_gifs_path=gifFolder,fps=5)
    
    
    backgroundDataPath = os.path.join(os.path.dirname(__file__), "intruder_background.npy")
    backgroundData = np.load(backgroundDataPath)

    modelpath = f"models/{JOBID}/SectorCREnv-v0/SectorCREnv-v0_TD3_singleEnv_baseline_model_mp.zip"
    model = TD3.load(modelpath, env=saliencyEnv,device='cpu')

    n_eps = 10
    max_steps = 50

    for i in range(n_eps):
        done = truncated = False
        obs, info = saliencyEnv.reset(options=options)
        step = 0
        
        while not (done or truncated) and step < max_steps:
            step+=1
            action, _states = model.predict(obs, deterministic=True)
            shap_values = runBackgroundExplainer(model, obs,backgroundData,intruder_feature_mapping,n_samples=300)

            obs, reward, done, truncated, info = saliencyEnv.step(action[()],shap_values)
            
        if step == max_steps:
            saliencyEnv.export_episode_gif()
            
            
    env.close()




