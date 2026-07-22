"""Example script how to calculate and visualize SHAP values for safe state explanations in SectorCR environment."""

import gymnasium as gym
from stable_baselines3 import SAC,TD3,DDPG,PPO
import bluesky_gym
import numpy as np
import logging
from bluesky_gym.wrappers.xrlMethods.state.saliency.sector_cr_env_saliency import SaliencySectorControl
from src.XRL.shapMethods.shap_explainers import runSafeStateExplainer


from bluesky_gym.utils import logger
bluesky_gym.register_envs()




DEBUG = False

D_NORTH = 0
D_EAST = 30000 

# 1. Set your normalized relative velocities
vx_r_norm = -0.5
vy_r_norm = 0.0

# 2. Denormalize to get the physical vector (using factors from sector_cr_env.py)
#    vx_r is divided by 32, vy_r is divided by 66 in the env
vx_r_raw = vx_r_norm * 32
vy_r_raw = vy_r_norm * 66

track_rad = np.arctan2(vy_r_raw, vx_r_raw)

SAFE_VALS = {
            "x_r": D_NORTH/13000,    # Behind
            "y_r": D_EAST/13000,     # Centered
            "vx_r": vx_r_norm,   # Flying away
            "vy_r": vy_r_norm,
            "cos(track)": np.cos(track_rad),
            "sin(track)": np.sin(track_rad),
            "distances": (np.sqrt(D_NORTH**2 + D_EAST**2)-50000)/15000
        }
    
if __name__ == "__main__":
    # Initialize the environment
    env_name = 'SectorCREnv-v0'
    JOBID = "4675598"
    SEED = 42
    color_mode = "default"  #"clipped"  #"scaled"
    if DEBUG:
        gifFolder = f"./plots/{JOBID}/{env_name}/shapSafeStateDebug/"
    else:
        gifFolder = f"./plots/{JOBID}/{env_name}/shapSafeState/{color_mode}/"
    # Spawns more intruders than original env
    spawnFactor = 2
    options = {"SpawnFactor":spawnFactor}
    
    logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
    env = gym.make(env_name, render_mode="human")
    env.reset(seed=SEED,options=options)
    
    modelpath = f"models/{JOBID}/{env_name}/SectorCREnv-v0_TD3_singleEnv_baseline_model_mp.zip"

    model = TD3.load(modelpath,device='cpu')

    saliencyEnv = SaliencySectorControl(env,SAFE_VALS,DEBUG,export_gifs_path=gifFolder,fps=5,color_mode=color_mode,plot_action_path=True,plot_safe_path=True,model=model)

    n_eps = 10
    max_steps = 50

    for i in range(n_eps):
        done = truncated = False
        obs, info = saliencyEnv.reset(options=options)
        step = 0
        
        while not (done or truncated) and step < max_steps:
            step+=1
            action, _states = model.predict(obs, deterministic=True)
            shap_values = runSafeStateExplainer(model, obs,SAFE_VALS)

            obs, reward, done, truncated, info = saliencyEnv.step(action[()],shap_values)
            
        if step == max_steps:
            saliencyEnv.export_episode_gif()
            
            
    env.close()


