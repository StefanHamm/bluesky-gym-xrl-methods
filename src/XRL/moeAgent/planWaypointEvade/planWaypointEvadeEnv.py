import gymnasium as gym
from stable_baselines3 import SAC,TD3,DDPG,PPO
import bluesky_gym
import numpy as np
from src.XRL.moeAgent.controlEvade import ThresholdGating, BlendingGating
bluesky_gym.register_envs()


def metric_extractor(obs):
    # Example: Use the minimum distance to intruders as the metric
    return np.min(obs["intruder_distance"])





if __name__ == "__main__":
    env_name = 'PlanWaypointEvadeEnv-v0'
    env = gym.make(env_name,render_mode='human')
    env.reset(seed=42)
    evade_env_name = 'FreeFlightCREnv-v0'
    control_env_name = 'PlanWaypointEnv-v2'
    
    
    
    control_modelpath = f"models/01/{control_env_name}/{control_env_name}_SAC_vecEnvLogs_baseline_model_mp.zip"
    control_model = SAC.load(control_modelpath,device='cpu')
    control_keywords = [
        "waypoint_distance",
        "cos_difference",
        "sin_difference",
        "waypoint_reached",
        "previous_action"
    ]
    
    evade_modelpath = f"models/01/{evade_env_name}/{evade_env_name}_SAC_vecEnvLogs_baseline_model_mp.zip"
    evade_model = SAC.load(evade_modelpath,device='cpu')
    evade_keywords = [
        "intruder_distance",
        "cos_difference_pos",
        "sin_difference_pos",
        "x_difference_speed",
        "y_difference_speed",
        "cos_own_heading",
        "sin_own_heading"
    ]
    
    print(control_model.observation_space)
    
    gated_model = ThresholdGating(
    controlModel=control_model, 
    evadeModel=evade_model,
    controlKeys=control_keywords,
    evadeKeys=evade_keywords,
    threshold=0.15, 
    metric_extractor=metric_extractor
)
    
    gated_model = BlendingGating(
    controlModel=control_model, 
    evadeModel=evade_model,
    controlKeys=control_keywords,
    evadeKeys=evade_keywords,
    min_val=0.05,
    max_val=0.2, 
    metric_extractor=metric_extractor
)
    
    
    episodes = 10
    for ep in range(episodes):
        done = truncated = False
        obs, info = env.reset()
        step = 0
        
        while not (done or truncated):
            step+=1
            action,evading = gated_model.predict(obs, deterministic=True)
            env.unwrapped.update_evading_status(evading)
            obs,reward,done,turncated,info = env.step(action)