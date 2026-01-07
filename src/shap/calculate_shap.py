"""
This file contains the example code used during the first part of the workshop.
Do not run the code directly from here, but instead, copy it from this file 
to the corresponding file as indicated in the workshop.
"""

import gymnasium as gym
from stable_baselines3 import SAC,TD3
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
#env = gym.make(env_name, render_mode=None)
# file_name = 'my_first_bsg_experiment.csv'
# logger = logger.CSVLoggerCallback('logs/', file_name)

#Train a model for 'n' timesteps
# model = SAC('MultiInputPolicy', env=env, verbose=1)
# model.learn(total_timesteps=2e6, callback=logger,progress_bar=True)
# model.save("models/SAC")
# env.close()

DEBUG = True

SAFE_VALS = {
            "dist": 10.0,
            "cos": -1.0,  # Behind
            "sin": 0.0,
            "dx": 0.0,    # Flying away
            "dy": -1.0
        }

def DEBUG_baselineObservation(observation):
    obs_copy = copy.deepcopy(observation)
    
    mean = False
    
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


def runPermutationExplainer(model, observation):
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
        
        predictions = []
        
        # Safe/Behind values
        safe_vals = {
            "dist": 10.0,
            "cos": -1.0,  # Behind
            "sin": 0.0,
            "dx": 0,    # Flying away
            "dy": -1.0
        }
        logging.debug(f"X_batch: {X_batch}")
        for row_indices in X_batch:
            # Create a fresh observation from the original
            obs_copy = copy.deepcopy(observation)
            
            # Loop through the indices to see which are "Real" and which are "Masked" (-1)
            for i, idx_val in enumerate(row_indices):
                if idx_val == 0:
                    # Apply your masking logic here!
                    obs_copy["intruder_distance"][i] = SAFE_VALS["dist"]
                    obs_copy["cos_difference_pos"][i] = SAFE_VALS["cos"]
                    obs_copy["sin_difference_pos"][i] = SAFE_VALS["sin"]
                    obs_copy["x_difference_speed"][i] = SAFE_VALS["dx"]
                    obs_copy["y_difference_speed"][i] = SAFE_VALS["dy"]
            
            # Predict
            # SB3 models return tuple (action, state), take [0]
            pred, _ = model.predict(obs_copy, deterministic=True)
            predictions.append(pred)
            
        return np.array(predictions)

    # 4. RUN
    # Note: We pass the WRAPPER as the model, and cheat_masker as the masker
    explainer = shap.explainers.Permutation(custom_model_wrapper, cheat_masker)
    
    shap_values = explainer(testX)
    #shap.plots.bar(shap_values)
    return shap_values
    

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    env = gym.make(env_name, render_mode="human")
    saliencyEnv = SaliencyHorizontalControl(env,SAFE_VALS,DEBUG)
    model = SAC.load("models/4675598/HorizontalCREnv-v0/HorizontalCREnv-v0_SAC_singleEnv_baseline_model_mp.zip", env=saliencyEnv,device='cpu')
    n_eps = 10
    for i in range(n_eps):
        done = truncated = False
        obs, info = saliencyEnv.reset()
        i = 0
        while not (done or truncated):
            i+=1
            
            if DEBUG:
                action, _states = model.predict(DEBUG_baselineObservation(obs), deterministic=True)
                logging.info(f"DEBUG action taken: {action}")
            else:
                action, _states = model.predict(obs, deterministic=True)
            shap_values = None
            logging.info(f"Action taken: {action}")
            
            if i % 1 == 0:
                logging.info(f"Episode {i+1} finished.")
                shap_values = runPermutationExplainer(model, obs)
                logging.info(f"shap_values: {shap_values}")
                
            obs, reward, done, truncated, info = saliencyEnv.step(action[()],shap_values)
            
    env.close()




