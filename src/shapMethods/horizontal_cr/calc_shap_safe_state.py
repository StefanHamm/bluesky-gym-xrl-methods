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
from bluesky_gym.wrappers.xrlMethods.state.horizontal_cr_env_saliency import SaliencyHorizontalControl

from bluesky_gym.utils import logger
bluesky_gym.register_envs()

# Initialize the environment and logger
env_name = 'HorizontalCREnv-v0'


DEBUG = False

SAFE_VALS = {
            "dist": 0.5,
            "cos": -1.0,  # Behind
            "sin": 0.0,
            "dx": 1.0,    # Flying away
            "dy": 1.0
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
            obs_copy["intruder_distance"][i] = SAFE_VALS["dist"]
            obs_copy["cos_difference_pos"][i] = SAFE_VALS["cos"]
            obs_copy["sin_difference_pos"][i] = SAFE_VALS["sin"]
            obs_copy["x_difference_speed"][i] = SAFE_VALS["dx"]
            obs_copy["y_difference_speed"][i] = SAFE_VALS["dy"]
    return obs_copy


def runExactExplainer(model, observation):
    # 1. SETUP: We tell SHAP to explain features 0, 1, 2... (the intruders)
    number_of_aircrafts = len(observation["intruder_distance"])
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
                obs_batch["intruder_distance"][i, masked_indices] = SAFE_VALS["dist"]
                obs_batch["cos_difference_pos"][i, masked_indices] = SAFE_VALS["cos"]
                obs_batch["sin_difference_pos"][i, masked_indices] = SAFE_VALS["sin"]
                obs_batch["x_difference_speed"][i, masked_indices] = SAFE_VALS["dx"]
                obs_batch["y_difference_speed"][i, masked_indices] = SAFE_VALS["dy"]
            
        # 3. Batch Predict
        # Single call for all permutations
        pred, _ = model.predict(obs_batch, deterministic=True)
            
        return np.array(pred)

    # 4. RUN
    # Note: We pass the WRAPPER as the model, and cheat_masker as the masker
    explainer = shap.explainers.Exact(custom_model_wrapper, cheat_masker)
    
    #shap_values = explainer(testX,max_evals=2*len(observation["intruder_distance"])+1)
    shap_values = explainer(testX)
    
    #shap.plots.bar(shap_values)
    return shap_values
    

if __name__ == "__main__":
    
    #JOBID = "4676447"
    JOBID = "4675598"
    SEED = 42
    EXPORT = True
    PRINT_ACTION_PATH = True
    PLOT_SAFE_PATH = True
    color_mode = "default"  #"clipped"  #"scaled"EX
    #plots/jobid/gifs/
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
    
    #modelpath = f"models/{JOBID}/HorizontalCREnv-v0_SafeObservationWrapper/HorizontalCREnv-v0_SAC_baseline_model_mp.zip"
    modelpath = f"models/{JOBID}/HorizontalCREnv-v0/HorizontalCREnv-v0_SAC_singleEnv_baseline_model_mp.zip"
    #model = PPO.load(modelpath, env=saliencyEnv,device='cpu')
    model = SAC.load(modelpath,device='cpu')
    #model = DDPG.load(modelpath)
    
    saliencyEnv = SaliencyHorizontalControl(env,SAFE_VALS,DEBUG,export_gifs_path=gifFolder,fps=5,color_mode=color_mode,plot_action_path=PRINT_ACTION_PATH,model=model,plot_safe_path=PLOT_SAFE_PATH)
    
    
    
   
    n_eps = 20
    for i in range(n_eps):
        done = truncated = False
        obs, info = saliencyEnv.reset()
        step = 0
        # skip first 10 epsiodes
        # if i < 10:
        #     continue
        while not (done or truncated):
            step+=1
            
            if DEBUG:
                action, _states = model.predict(DEBUG_baselineObservation(obs), deterministic=True)
                logging.info(f"DEBUG action taken: {action}")
            else:
                action, _states = model.predict(obs, deterministic=True)
            shap_values = None
            logging.info(f"Action taken: {action}")
            
            if step % 1 == 0:
                logging.info(f"Episode {i+1} finished.")
                shap_values = runExactExplainer(model, obs)
                
                logging.info(f"shap_values: {shap_values}")
                
                
            obs, reward, done, truncated, info = saliencyEnv.step(action[()],shap_values)
            
    env.close()



