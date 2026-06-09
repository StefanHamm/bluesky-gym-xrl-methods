import gymnasium as gym
from stable_baselines3 import SAC
import bluesky_gym
import argparse
import logging
from bluesky_gym.wrappers.xrlMethods.state.saliency.vertical_cr_env_saliency import SaliencyVerticalControl
from src.XRL.shapMethods.shap_explainers import runSafeStateExplainer

bluesky_gym.register_envs()

env_name = 'VerticalCREnv-v0'

SAFE_VALS = {
    "intruder_distance": 1.0,
    "cos_difference_pos": -1.0,
    "sin_difference_pos": 0.0,
    "altitude_difference": 1.0,
    "x_difference_speed": 1.0,
    "y_difference_speed": 1.0,
    "z_difference_speed": 0.0
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_eps', type=int, default=5)
    parser.add_argument('--max_steps', type=int, default=50)
    parser.add_argument('--model_path', type=str, required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
    
    env = gym.make(env_name, render_mode="human")
    
    model = SAC.load(args.model_path, device='cpu')

    saliencyEnv = SaliencyVerticalControl(env, SAFE_VALS, fps=5, color_mode="clipped", model=model)
 
    for i in range(args.n_eps):
        done = truncated = False
        obs, info = saliencyEnv.reset(seed=42+i)
        step = 0
        
        while not (done or truncated) and step < args.max_steps:
            step += 1
            action, _states = model.predict(obs, deterministic=True)
            shap_values = runSafeStateExplainer(model, obs, SAFE_VALS)
            obs, reward, done, truncated, info = saliencyEnv.step(action[()], shap_values)
            
    env.close()
