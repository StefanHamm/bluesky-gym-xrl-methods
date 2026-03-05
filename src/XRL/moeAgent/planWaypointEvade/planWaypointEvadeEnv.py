import gymnasium as gym
from stable_baselines3 import SAC,TD3,DDPG,PPO
import bluesky_gym
import numpy as np
from src.XRL.moeAgent.controlEvade import ThresholdGating, BlendingGating, FatigueBlendingGating, PredictiveShieldingGating
from bluesky_gym.envs.free_flight_env import SENSOR_RANGE
from bluesky_gym.utils.constants import NM2KM

bluesky_gym.register_envs()


def metric_extractor(obs):
    # Example: Use the minimum distance to intruders as the metric
    return np.min(obs["intruder_distance"])





if __name__ == "__main__":
    env_name = 'PlanWaypointEvadeEnv-v0'
    env = gym.make(env_name,render_mode='human',training=False)
    env.reset(seed=42)
    evade_env_name = 'FreeFlightCREnv-v0'
    control_env_name = 'PlanWaypointEnv-v2'
    
    
    
    control_modelpath = r"models\4901832\PlanWaypointEnv-v2\PlanWaypointEnv-v2_SAC_vecEnvLogs_baseline_model_mp.zip"
    control_model = SAC.load(control_modelpath,device='cpu')
    control_keywords = [
        "waypoint_distance",
        "cos_difference",
        "sin_difference",
        "waypoint_reached",
        "previous_action"
    ]
    
    evade_modelpath = r"models\4901832\FreeFlightCREnv-v0\FreeFlightCREnv-v0_SAC_vecEnvLogs_baseline_model_mp.zip"
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
    
    min_nm = 7
    max_nm = 15
    
    
    
    min_normed = min_nm * NM2KM / SENSOR_RANGE
    max_normed = max_nm * NM2KM / SENSOR_RANGE
    
    
    
    gated_model = ThresholdGating(
    controlModel=control_model, 
    evadeModel=evade_model,
    controlKeys=control_keywords,
    evadeKeys=evade_keywords,
    threshold=min_normed, 
    metric_extractor=metric_extractor
)
    
    
 
    gated_model = BlendingGating(
    controlModel=control_model, 
    evadeModel=evade_model,
    controlKeys=control_keywords,
    evadeKeys=evade_keywords,
    min_val=min_normed,  # sensor range is 250 NM
    max_val=max_normed, 
    metric_extractor=metric_extractor
)
    
#     gated_model = FatigueBlendingGating(
#     controlModel=control_model, 
#     evadeModel=evade_model,
#     controlKeys=control_keywords,
#     evadeKeys=evade_keywords,
#     min_val=min_normed,  # sensor range is 250 NM
#     max_val=max_normed, 
#     fatigue_rate=0.005,
#     max_fatigue=0.3,
#     metric_extractor=metric_extractor
# )

#     gated_model = PredictiveShieldingGating(
#     controlModel=control_model, 
#     evadeModel=evade_model,
#     controlKeys=control_keywords,
#     evadeKeys=evade_keywords,
#     number_of_future_steps=240, # lookahead in seconds since step is 1 second
#     number_intrusor_aircraft=5,
#     intrusion_distance_nm=5,
#     alpha_update_interval=5 # every n steps the future prediction is performed
# )
    
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