import gymnasium as gym
from stable_baselines3 import SAC
import bluesky_gym
import argparse
import shap
import logging
import torch
import random
import numpy as np
from bluesky_gym.wrappers.xrlMethods.state.saliency.descent_env_saliency import SaliencyDescentControl
from src.XRL.shapMethods.shap_explainers import get_background_data_descent, runFeatureBackgroundExplainer

bluesky_gym.register_envs()
env_name = 'DescentEnv-v0'

def set_deterministic_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_eps', type=int, default=3)
    parser.add_argument('--max_steps', type=int, default=50)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--bg_samples', type=int, default=500)
    parser.add_argument('--k_centroids', type=int, default=15)
    parser.add_argument("--export_path", type=str, default="./replays", help="Path to save visual replays")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Enforce global determinism
    set_deterministic_seed(args.seed)
    
    model = SAC.load(args.model_path, device='cpu')

    print("Phase 1: Collecting background trajectory data...")
    bg_env = gym.make(env_name)
    # Ensure the background environment is explicitly seeded on its first reset
    bg_env.action_space.seed(args.seed)
    bg_env.observation_space.seed(args.seed)
    
    bg_data = get_background_data_descent(bg_env, model, n_samples=args.bg_samples, seed=args.seed)
    bg_env.close()

    print(f"Phase 2: Summarizing to {args.k_centroids} K-Means centroids...")
    background_summary = shap.kmeans(bg_data, args.k_centroids)

    print("Phase 3: Starting visual XRL loop...")
    env = gym.make(env_name, render_mode="human")
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)

    saliencyEnv = SaliencyDescentControl(env, fps=5, model=model, export_gifs_path=args.export_path)
 
    for i in range(args.n_eps):
        obs, _ = saliencyEnv.reset(seed=42+i)
        step = 0
        done = truncated = False
        
        while not (done or truncated) and step < args.max_steps:
            step += 1
            action, _ = model.predict(obs, deterministic=True)
            
            # Execute feature-level KernelSHAP
            shap_values = runFeatureBackgroundExplainer(model, obs, background_summary)
            
            obs, reward, done, truncated, _ = saliencyEnv.step(action[()], shap_values)
            
    env.close()
