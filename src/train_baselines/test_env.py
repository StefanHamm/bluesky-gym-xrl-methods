import argparse
import gymnasium as gym
from stable_baselines3 import SAC
import bluesky_gym

bluesky_gym.register_envs()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a trained SAC policy in BlueSky-Gym")
    parser.add_argument("--env_name", type=str, required=True, help="Target environment ID (e.g., VerticalCREnv-v0)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained .zip model checkpoint")
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes")
    args = parser.parse_args()

    env = gym.make(args.env_name, render_mode="human") 

    model = SAC.load(args.model_path)

    for i in range(args.episodes):
        obs, info = env.reset()
        total_reward = 0
        
        for _ in range(500):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                reason = "Reached Waypoint" if info.get('waypoint_reached', 0) == 1 else \
                         "Crashed" if info.get('crashed', 0) == 1 else \
                         "Truncated (Time limit)" if truncated else \
                         "Out of Bounds"
                break
                
        print(f"Episode {i+1} total reward: {total_reward:.2f} | Result: {reason} | Info: {info}")

    env.close()
