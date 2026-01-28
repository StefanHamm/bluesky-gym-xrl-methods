import gymnasium as gym
import numpy as np
import copy
import pickle
from stable_baselines3 import SAC,TD3
import bluesky_gym
import bluesky_gym.envs
from tqdm import tqdm
import os

# Environment and model setup (copied from calculate_shap_safe_state.py)
env_name = 'SectorCREnv-v0'
JOBID = "4675598"

# Register environments
bluesky_gym.register_envs()

env = gym.make(env_name, render_mode=None)

modelpath = f"models/{JOBID}/SectorCREnv-v0/SectorCREnv-v0_TD3_singleEnv_baseline_model_mp.zip"
model = TD3.load(modelpath, env=env, device='cpu')

# Run the simulation n episodes with rendering off. For each n steps in the episode collect a random intruder from the observation. Save its variables in a list.
# This builds a background distribution of possible intruder states.

def collect_intruder_background(env, model, n_episodes=50, steps_per_episode=100, sample_every=1):
    rows = []
    for ep in tqdm(range(n_episodes), desc="Episodes"):
        obs, info = env.reset()
        done = truncated = False
        step = 0
        while not (done or truncated) and step < steps_per_episode:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action[()])
            if step % sample_every == 0:
                n_intruders = len(obs["distances"])
                idx = np.random.randint(n_intruders)
                # Each row: [dist, cos, sin, dx, dy]
                row = [
                    obs["x_r"][idx],
                    obs["y_r"][idx],
                    obs["vx_r"][idx],
                    obs["vy_r"][idx],
                    obs["cos(track)"][idx],
                    obs["sin(track)"][idx],
                    obs["distances"][idx]
                ]
                rows.append(row)
            step += 1
    return np.array(rows)

if __name__ == "__main__":
    sample_every = 5  # Example: sample every 5 steps
    background = collect_intruder_background(env, model, n_episodes=300, steps_per_episode=200, sample_every=sample_every)
    # Save as numpy array for easy export in the same folder as this script
    out_path = os.path.join(os.path.dirname(__file__), "intruder_background.npy")
    np.save(out_path, background)
    print(f"Collected {background.shape[0]} intruder states. Shape: {background.shape}")
    print(f"Saved to {out_path}")
