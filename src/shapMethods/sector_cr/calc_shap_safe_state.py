"""
This file contains the example code used during the first part of the workshop.
Do not run the code directly from here, but instead, copy it from this file 
to the corresponding file as indicated in the workshop.
"""

import gymnasium as gym
from stable_baselines3 import SAC,TD3,DDPG,PPO
import bluesky_gym
import bluesky_gym.envs
import shap
import numpy as np
import matplotlib.pyplot as plt
import logging
import copy
from bluesky_gym.wrappers.xrlMethods.state.sector_cr_env_saliency import SaliencySectorControl

from bluesky_gym.utils import logger
bluesky_gym.register_envs()

# Initialize the environment and logger
env_name = 'SectorCREnv-v0'


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

def DEBUG_baselineObservation(observation):
    obs_copy = copy.deepcopy(observation)
    
    mean = False
    logging.info(f"Original Observation: {obs_copy}")
    
    if mean:
        for k in obs_copy:
            obs_copy[k][:] = np.mean(obs_copy[k])
        return obs_copy
    
    else:
        for i in range(len(obs_copy["intruder_distance"])): 
            obs_copy["x_r"][i] = SAFE_VALS["x_r"]
            obs_copy["y_r"][i] = SAFE_VALS["y_r"]
            obs_copy["vx_r"][i] = SAFE_VALS["vx_r"]
            obs_copy["vy_r"][i] = SAFE_VALS["vy_r"]
            obs_copy["cos(track)"][i] = SAFE_VALS["cos(track)"]
            obs_copy["sin(track)"][i] = SAFE_VALS["sin(track)"]
            obs_copy["distances"][i] = SAFE_VALS["distances"]
    return obs_copy


def runExactExplainer(model, observation):
    # 1. SETUP: We tell SHAP to explain features 0, 1, 2... (the intruders)
    number_of_aircrafts = len(observation["distances"])
    # We pass indices [0, 1, 2...] as the "Input" to SHAP
    testX = np.array([np.arange(number_of_aircrafts)]) 

    # 2. MASKER: Returns simple modified INDICES
    # This keeps SHAP happy because it gets arrays it can concatenate.
    def cheat_masker(mask, X):
        # X is just [0, 1, 2...], mask is [True, False, True...]
        
        # We use a special flag (-1) to mark "masked" intruders
        # masked_X = X.copy()
        # masked_X[~mask] = -1 
        logging.debug(f"mask: {mask}")
        return [mask]

    # 3. MODEL WRAPPER: The "Real" Masker
    # This intercepts the array from SHAP, builds the dictionary, and calls your model.
    def custom_model_wrapper(X_batch):
        #print(len(X_batch))
        # X_batch is a 2D array of indices, e.g.:
        # [[0, 1, 2],
        #  [-1, 1, 2],  <-- Intruder 0 is masked here
        #  [0, -1, 2]]
        total_evals = len(X_batch)
        
        # Batch for efficiency
        # 1. Create a large batch of observations by replicating the original one
        obs_batch = {k: np.tile(v, (total_evals, 1)) for k, v in observation.items()}
        
       
        logging.debug(f"X_batch: {X_batch}")
        
        # 2. Vectorized or efficient update of the batch
        # Iterate over each mask (row in X_batch) and update corresponding obs in obs_batch
        for i, row_indices in enumerate(X_batch):
            # row_indices is the mask for the i-th observation in the batch
            # Find indices where intruder should be masked (value is 0/False)
            masked_indices = np.where(row_indices==0)[0]
            
            if len(masked_indices) > 0:
                obs_batch["x_r"][i, masked_indices] = SAFE_VALS["x_r"]
                obs_batch["y_r"][i, masked_indices] = SAFE_VALS["y_r"]
                obs_batch["vx_r"][i, masked_indices] = SAFE_VALS["vx_r"]
                obs_batch["vy_r"][i, masked_indices] = SAFE_VALS["vy_r"]
                obs_batch["cos(track)"][i, masked_indices] = SAFE_VALS["cos(track)"]
                obs_batch["sin(track)"][i, masked_indices] = SAFE_VALS["sin(track)"]
                obs_batch["distances"][i, masked_indices] = SAFE_VALS["distances"]
            
        # 3. Batch Predict
        # Single call for all permutations
        pred, _ = model.predict(obs_batch, deterministic=True)
       
        return np.array(pred)

    # 4. RUN
    # Note: We pass the WRAPPER as the model, and cheat_masker as the masker
    explainer = shap.explainers.Exact(custom_model_wrapper, cheat_masker)
    
    shap_values = explainer(testX)
    #shap.plots.bar(shap_values)
    return shap_values
    

if __name__ == "__main__":
    
    JOBID = "4676447" # VEC env SAC
    #JOBID = "4706792"
    SEED = 42
    color_mode = "default"  #"clipped"  #"scaled"
    #plots/jobid/gifs/
    if DEBUG:
        gifFolder = f"./plots/{JOBID}/{env_name}/shapSafeStateDebug/"
    else:
        gifFolder = f"./plots/{JOBID}/{env_name}/shapSafeState/{color_mode}/"
    spawnFactor = 2
    options = {"SpawnFactor":spawnFactor}
    
    logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
    env = gym.make(env_name, render_mode="human")
    env.reset(seed=SEED,options=options)

    
    saliencyEnv = SaliencySectorControl(env,SAFE_VALS,DEBUG,export_gifs_path=gifFolder,fps=5,color_mode=color_mode)
    
    
    
    #modelpath = f"models/{JOBID}/HorizontalCREnv-v0_SafeObservationWrapper/HorizontalCREnv-v0_SAC_baseline_model_mp.zip"
    modelpath = f"models/{JOBID}/SectorCREnv-v0/SectorCREnv-v0_SAC_vecEnvLogs_baseline_model_mp.zip"
    #model = PPO.load(modelpath, env=saliencyEnv,device='cpu')
    model = SAC.load(modelpath, env=saliencyEnv,device='cpu')
    #model = DDPG.load(modelpath)
    max_steps = 50
    n_eps = 6
    for i in range(n_eps):
        done = truncated = False
        obs, info = saliencyEnv.reset(options=options)
        step = 0

        while not (done or truncated) and step < max_steps:
            step+=1
            
            if DEBUG:
                #action, _states = model.predict(DEBUG_baselineObservation(obs), deterministic=True)
                action, _states = model.predict(obs, deterministic=True)
                logging.info(f"DEBUG action taken: {action}")
            else:
                action, _states = model.predict(obs, deterministic=True)
            shap_values = None
            logging.info(f"Action taken: {action}")
            
            if step % 1 == 0:
                logging.info(f"Episode {i+1} finished.")
                shap_values = runExactExplainer(model, obs)
                logging.info(f"shap_values: {shap_values}")
                
                
         
            #obs, reward, done, truncated, info = saliencyEnv.step(action[()],shap_values)
            obs, reward, done, truncated, info = saliencyEnv.step(action[()],shap_values=shap_values)
        if step == max_steps:
            saliencyEnv.export_gif()
            
    env.close()



