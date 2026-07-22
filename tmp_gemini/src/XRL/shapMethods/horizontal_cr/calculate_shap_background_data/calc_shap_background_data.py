"""Example script how to calculate and visualize SHAP values for background data explanations in HorizontalCR environment."""
import gymnasium as gym
from stable_baselines3 import SAC,TD3,DDPG,PPO
import bluesky_gym
import numpy as np
import logging
from bluesky_gym.wrappers.xrlMethods.state.saliency.horizontal_cr_env_saliency import SaliencyHorizontalControl
from src.XRL.shapMethods.shap_explainers import runBackgroundExplainer
import numpy as np

import os
bluesky_gym.register_envs()

# Initialize the environment and logger
env_name = 'HorizontalCREnv-v0'


if __name__ == "__main__":
    
    intruder_feature_mapping = [
        "intruder_distance","cos_difference_pos","sin_difference_pos",
        "x_difference_speed","y_difference_speed"]
    
    JOBID = "4675598"
    SEED = 42


    DEBUG = False
    RUN_BASELINE_ACTION = False
    EXPORT = False
    
    gifFolder= None

    color_mode = "default"  #"clipped"  #"scaled"
    if EXPORT: 
        if DEBUG:
            gifFolder = f"./plots/{JOBID}/{env_name}/shapBackgroundDataDebug/"
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        else:
            gifFolder = f"./plots/{JOBID}/{env_name}/shapBackgroundData/{color_mode}/"
        
            logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
    
    env = gym.make(env_name, render_mode="human")
    env.reset(seed=SEED)
    saliencyEnv = SaliencyHorizontalControl(env,None,None,export_gifs_path=gifFolder,fps=5)
    
    
    backgroundDataPath = os.path.join(os.path.dirname(__file__), "intruder_background.npy")
    backgroundData = np.load(backgroundDataPath)
    
    modelpath = f"models/{JOBID}/HorizontalCREnv-v0/HorizontalCREnv-v0_SAC_singleEnv_baseline_model_mp.zip"
    model = SAC.load(modelpath, env=saliencyEnv,device='cpu')
    
    n_eps = 10
    max_steps = 50

    for i in range(n_eps):
        done = truncated = False
        obs, info = saliencyEnv.reset()
        step = 0
        
        while not (done or truncated) and step < max_steps:
            step+=1
            action, _states = model.predict(obs, deterministic=True)
            shap_values = runBackgroundExplainer(model, obs,backgroundData,intruder_feature_mapping,n_samples=300)

            obs, reward, done, truncated, info = saliencyEnv.step(action[()],shap_values)
            
        if step == max_steps:
            saliencyEnv.export_episode_gif()
            
    env.close()




