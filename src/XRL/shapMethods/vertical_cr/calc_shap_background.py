import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
import bluesky_gym
import argparse
import logging
from bluesky_gym.wrappers.xrlMethods.state.saliency.vertical_cr_env_saliency import SaliencyVerticalControl
from src.XRL.shapMethods.shap_explainers import runBackgroundExplainer

bluesky_gym.register_envs()

env_name = 'VerticalCREnv-v0'

# The mapping keys dictate which observation features belong to the intruders
MAPPING_KEYS = [
    "intruder_distance",
    "cos_difference_pos",
    "sin_difference_pos",
    "altitude_difference",
    "x_difference_speed",
    "y_difference_speed",
    "z_difference_speed"
]

def get_background_data_dict(env, model, n_samples=1000):
    """Collects empirical background data, maintaining the dictionary observation structure."""
    obs_list = []
    obs, _ = env.reset()
    for _ in range(n_samples):
        # Extract features for all intruders
        flat_features = [obs[key] for key in MAPPING_KEYS]
        
        # Stack into (num_intruders, num_features)
        obs_list.append(np.column_stack(flat_features))
        
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
            
    # vstack flattens the list of 2D arrays into a single large 2D matrix: 
    # Shape: (n_samples * num_intruders, num_features)
    return np.vstack(obs_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_eps', type=int, default=5)
    parser.add_argument('--max_steps', type=int, default=50)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--bg_samples', type=int, default=1000)
    parser.add_argument("--export_path", type=str, default="./replays", help="Path to save visual replays")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
    
    model = SAC.load(args.model_path, device='cpu')

    # Phase 1: Collect empirical background distribution 
    print("Collecting background trajectory data...")
    bg_env = gym.make(env_name)
    background_data = get_background_data_dict(bg_env, model, n_samples=args.bg_samples)
    bg_env.close()

    # Phase 2: Execute Visual Saliency Loop
    print("Starting visual XRL evaluation...")
    env = gym.make(env_name, render_mode="human")
    
    saliencyEnv = SaliencyVerticalControl(env, safe_vals=None, fps=5, color_mode="clipped", model=model, export_gifs_path=args.export_path)
 
    for i in range(args.n_eps):
        done = truncated = False
        obs, info = saliencyEnv.reset(seed=42+i)
        step = 0
        
        while not (done or truncated) and step < args.max_steps:
            step += 1
            action, _states = model.predict(obs, deterministic=True)
            
            # Execute your existing entity-level Permutation/Exact SHAP Explainer
            shap_values = runBackgroundExplainer(
                model=model, 
                observation=obs, 
                backgroundData=background_data, 
                mapping=MAPPING_KEYS, 
                n_samples=50
            )
            
            obs, reward, done, truncated, info = saliencyEnv.step(action[()], shap_values)
            
    env.close()
