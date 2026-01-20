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
import numpy as np


from bluesky_gym.utils import logger
import os
bluesky_gym.register_envs()

# Initialize the environment and logger
env_name = 'HorizontalCREnv-v0'


def runExactExplainer(model, observation,backgroundData,n_samples=50):
    # runs shap permutation explainer on the given observation
    # returns shap values for the observation
    # this is done using the background data to sample from to displace  single intruders in the observation
    
    number_of_intruders = len(observation["intruder_distance"])
    # We pass indices [0, 1, 2...] as the "Input" to SHAP
    testX = np.array([np.arange(number_of_intruders)]) 

    # Masker: Returns multiple copies of the mask to allow averaging over background samples
    def cheat_masker(mask,X):
        return [mask]

    # Model Wrapper
    def custom_model_wrapper(X_batch):
        #print(len(X_batch))
        # Optimized: Batched creation and prediction
        n_masks = len(X_batch)
        total_evals = n_masks * n_samples
        
        # 1. Create a batch of observations by repeating the original observation
        # obs_batch = {key: (total_evals, features...)}
        obs_batch = {k: np.tile(v, (total_evals, 1)) for k, v in observation.items()}

        # 2. Vectorized filling of background data
        for m, mask_row in enumerate(X_batch):
            # Find indices of intruders that are masked (False)
            masked_indices = np.where(~np.array(mask_row, dtype=bool))[0]
            
            if len(masked_indices) > 0:
                # Determine the slice of rows corresponding to this mask
                start_idx = m * n_samples
                end_idx = start_idx + n_samples
                
                for intruder_idx in masked_indices:
                    # Generate random indices for background samples
                    rand_idxs = np.random.randint(0, len(backgroundData), size=n_samples)
                    samples = backgroundData[rand_idxs]
                    
                    # Fill the specific columns for this intruder across all n_samples
                    obs_batch["intruder_distance"][start_idx:end_idx, intruder_idx] = samples[:, 0]
                    obs_batch["cos_difference_pos"][start_idx:end_idx, intruder_idx] = samples[:, 1]
                    obs_batch["sin_difference_pos"][start_idx:end_idx, intruder_idx] = samples[:, 2]
                    obs_batch["x_difference_speed"][start_idx:end_idx, intruder_idx] = samples[:, 3]
                    obs_batch["y_difference_speed"][start_idx:end_idx, intruder_idx] = samples[:, 4]
        
        # 3. Batch prediction (Single call to model, much faster)
        preds, _ = model.predict(obs_batch, deterministic=True)
        
        # 4. Reshape and Average
        # preds shape: (total_evals, output_dim) -> reshape to (n_masks, n_samples, output_dim)
        # Then average over the n_samples dimension to get expected value per mask
        preds_reshaped = preds.reshape(n_masks, n_samples, -1)
        return preds_reshaped.mean(axis=1)

    explainer = shap.explainers.Exact(custom_model_wrapper, cheat_masker)
    
    shap_values = explainer(testX)
    return shap_values
    

if __name__ == "__main__":
    
    #JOBID = "4676447"
    JOBID = "4675598"
    SEED = 42
    #plots/jobid/gifs/
    DEBUG = False
    RUN_BASELINE_ACTION = False

    color_mode = "default"  #"clipped"  #"scaled"
    if DEBUG:
        gifFolder = f"./plots/{JOBID}/shapBackgroundDataDebug/"
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    else:
        gifFolder = f"./plots/{JOBID}/shapBackgroundData/{color_mode}/"
    
        logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
    env = gym.make(env_name, render_mode="human")
    env.reset(seed=SEED)
    saliencyEnv = SaliencyHorizontalControl(env,None,None,export_gifs_path=gifFolder,fps=5)
    
    
    backgroundDataPath = os.path.join(os.path.dirname(__file__), "intruder_background.npy")
    backgroundData = np.load(backgroundDataPath)
    #modelpath = f"models/{JOBID}/HorizontalCREnv-v0_SafeObservationWrapper/HorizontalCREnv-v0_SAC_baseline_model_mp.zip"
    modelpath = f"models/{JOBID}/HorizontalCREnv-v0/HorizontalCREnv-v0_SAC_singleEnv_baseline_model_mp.zip"
    #model = PPO.load(modelpath, env=saliencyEnv,device='cpu')
    model = SAC.load(modelpath, env=saliencyEnv,device='cpu')
    #model = DDPG.load(modelpath)
    n_eps = 6
    for i in range(n_eps):
        done = truncated = False
        obs, info = saliencyEnv.reset()
        step = 0
        
        while not (done or truncated):
            step+=1
            

            action, _states = model.predict(obs, deterministic=True)
            shap_values = None
            logging.info(f"Action taken: {action}")
            
            if step % 1 == 0:
                logging.info(f"Episode {i+1} finished.")
                shap_values = runExactExplainer(model, obs,backgroundData,n_samples=300)
                logging.info(f"shap_values: {shap_values}")
                
            
            if DEBUG and RUN_BASELINE_ACTION:
                action = shap_values.base_values[0]
                logging.info(f"Overriding action with baseline action: {action}")
                
                obs, reward, done, truncated, info = saliencyEnv.step(action,shap_values)
            else:
                obs, reward, done, truncated, info = saliencyEnv.step(action[()],shap_values)
            
            
            
            
    env.close()




