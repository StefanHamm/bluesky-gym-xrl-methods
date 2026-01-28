"""Example script how to calculate and visualize SHAP values for safe state explanations in HorizontalCR environment."""

import gymnasium as gym
from stable_baselines3 import SAC,TD3,DDPG,PPO
import bluesky_gym
import shap
import numpy as np
import logging
from bluesky_gym.wrappers.xrlMethods.state.saliency.horizontal_cr_env_saliency import SaliencyHorizontalControl
from src.XRL.shapMethods.shap_explainers import runSafeStateExplainer

bluesky_gym.register_envs()

# Initialize the environment and logger
env_name = 'HorizontalCREnv-v0'


DEBUG = False

SAFE_VALS = {
            "intruder_distance": 0.5,
            "cos_difference_pos": -1.0,  # Behind
            "sin_difference_pos": 0.0,
            "x_difference_speed": 1.0,    # Flying away
            "y_difference_speed": 1.0
        }


if __name__ == "__main__":
    

    JOBID = "4675598"
    SEED = 42
    EXPORT = True
    PRINT_ACTION_PATH = True
    PLOT_SAFE_PATH = True
    color_mode = "default"  #"clipped"  #"scaled"EX

    gifFolder= None
    if EXPORT:
        if DEBUG:
            gifFolder = f"./plots/{JOBID}/{env_name}/shapSafeStateDebug/"
        elif PRINT_ACTION_PATH: 
            gifFolder = f"./plots/{JOBID}/{env_name}/withActionPath/shapSafeState/{color_mode}/"
        else:
            
            gifFolder = f"./plots/{JOBID}/{env_name}/shapSafeState/{color_mode}/"
    
    logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
    env = gym.make(env_name, render_mode="human")
    env.reset(seed=SEED)
    
    
    modelpath = f"models/{JOBID}/HorizontalCREnv-v0/HorizontalCREnv-v0_SAC_singleEnv_baseline_model_mp.zip"
    model = SAC.load(modelpath,device='cpu')

    
    saliencyEnv = SaliencyHorizontalControl(env,SAFE_VALS,DEBUG,export_gifs_path=gifFolder,fps=5,color_mode=color_mode,plot_action_path=PRINT_ACTION_PATH,model=model,plot_safe_path=PLOT_SAFE_PATH)
 
    n_eps = 10
    max_steps = 50

    for i in range(n_eps):
        done = truncated = False
        obs, info = saliencyEnv.reset()
        step = 0
        
        while not (done or truncated) and step < max_steps:
            step+=1
            action, _states = model.predict(obs, deterministic=True)
            shap_values = runSafeStateExplainer(model, obs,SAFE_VALS)

            obs, reward, done, truncated, info = saliencyEnv.step(action[()],shap_values)
            
        if step == max_steps:
            saliencyEnv.export_episode_gif()
            
    env.close()



