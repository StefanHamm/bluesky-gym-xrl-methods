import gymnasium as gym
from stable_baselines3 import SAC, PPO, TD3, A2C, DDPG
import bluesky_gym
import argparse
import shap
import logging
import torch
import random
import numpy as np
from bluesky_gym.wrappers.xrlMethods.state.saliency.static_obstacle_env_saliency import SaliencyStaticControl
from src.XRL.shapMethods.shap_explainers import get_background_data_static, runStaticObstacleExplainer

bluesky_gym.register_envs()
env_name = 'StaticObstacleEnv-v0'
#
# Map string arguments to SB3 algorithm classes
ALGO_MAP = {
    "SAC": SAC,
    "PPO": PPO,
    "TD3": TD3,
    "A2C": A2C,
    "DDPG": DDPG
}

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
    parser.add_argument('--algo', type=str, default='SAC', choices=['SAC', 'PPO', 'TD3', 'A2C', 'DDPG'], help="SB3 algorithm used for training")
    parser.add_argument('--bg_samples', type=int, default=500)
    parser.add_argument("--export_path", type=str, default="./replays", help="Path to save visual replays")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
    
    set_deterministic_seed(args.seed)

    AlgorithmClass = ALGO_MAP[args.algo]
    model = AlgorithmClass.load(args.model_path, device='cpu')
    #model = SAC.load(args.model_path, device='cpu')

    print("Phase 1: Collecting background trajectory data...")
    bg_env = gym.make(env_name)
    bg_env.action_space.seed(args.seed)
    bg_env.observation_space.seed(args.seed)
    
    bg_data = get_background_data_static(bg_env, model, seed=args.seed, n_samples=args.bg_samples)
    bg_env.close()

    print("Phase 2: Starting visual XRL loop...")
    env = gym.make(env_name, render_mode="human")
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)
    
    saliencyEnv = SaliencyStaticControl(env, fps=5, model=model, export_gifs_path=args.export_path)
 
    for i in range(args.n_eps):
        obs, _ = saliencyEnv.reset(seed=args.seed + i)
        step = 0
        done = truncated = False
        
        while not (done or truncated) and step < args.max_steps:
            step += 1
            action, _ = model.predict(obs, deterministic=True)
            
            # Execute entity-level Exact SHAP (outputs both Heading and Speed logic)
            shap_values = runStaticObstacleExplainer(model, obs, bg_data)
            
            obs, reward, done, truncated, _ = saliencyEnv.step(action[()], shap_values)
            
    env.close()
