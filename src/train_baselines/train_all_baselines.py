"""
This file is an example train and test loop for the different environments that
uses multiprocessing through the use of vectorised environments.
Note that multiprocessing doesn't necessarily result in faster training. It is
highly dependent on the environment and algorithm combination. If the algorithm
is able to train over a batch of observations, multiprocessing should lead to
faster training.
Selecting different environments is done through setting the 'env_name' variable.
"""

import gymnasium as gym
from stable_baselines3 import SAC,PPO,TD3,DDPG
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

import bluesky_gym
import bluesky_gym.envs

from bluesky_gym.utils import logger

bluesky_gym.register_envs()

#env_name = 'SectorCREnv-v0'

all_envs = ["SectorCREnv-v0","HorizontalCREnv-v0","StaticObstacleEnv-v0","PlanWaypointEnv-v0"]
algorithms = [SAC, PPO, TD3, DDPG, RecurrentPPO]
algorithm = SAC
num_cpu = 2



def make_env():
    """
    Utility function for multiprocessed env.
    """
    global env_counter
    env = gym.make(env_name, 
            render_mode=None)
    # Set a different seed for each created environment.
    env.reset(seed=env_counter)
    env_counter +=1 
    return env

# Initialize logger
# log_dir = f'./logs/{env_name}/'
# file_name = f'{env_name}_{str(algorithm.__name__)}.csv'
# csv_logger_callback = logger.CSVLoggerCallback(log_dir, file_name)

TRAIN = True
EVAL_EPISODES = 10
TOTAL_TIMESTEPS = 1e2
# Initialise the environment counter
env_counter = 0



if __name__ == "__main__":
    for env_name in all_envs:
        for algorithm in algorithms:
            # setup the logger for each environment-algorithm combination
            log_dir = f'./logs/{env_name}/'
            file_name = f'{env_name}_{str(algorithm.__name__)}_baseline.csv'
            csv_logger_callback = logger.CSVLoggerCallback(log_dir, file_name)
            env_counter = 0
            if TRAIN:
                # Create the vectorized environment
                env = make_vec_env(make_env, 
                        n_envs = num_cpu,
                        vec_env_cls=SubprocVecEnv)
                
                if algorithm == RecurrentPPO:
                    policy_type = "MultiInputLstmPolicy"
                else:
                    policy_type = "MultiInputPolicy"
                
                model = algorithm(policy_type, env, verbose=1,learning_rate=3e-4)
                model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=csv_logger_callback,progress_bar=True)
                model.save(f"models/{env_name}/{env_name}_{str(algorithm.__name__)}/baseline_model_mp")
                del model
                env.close()
                del env
    
    
    
    
    # env = make_vec_env(make_env, 
    #         n_envs = num_cpu,
    #         vec_env_cls=SubprocVecEnv)
    # model = algorithm("MultiInputPolicy", env, verbose=1,learning_rate=3e-4)
    # if TRAIN:
    #     model.learn(total_timesteps=1e3, callback=csv_logger_callback,progress_bar=True)
    #     model.save(f"models/{env_name}/{env_name}_{str(algorithm.__name__)}/model_mp")
    #     del model
    # env.close()
    # del env
    
    
    
    
    
    
    # # Test the trained model
    # env = gym.make(env_name, render_mode="human")
    # model = algorithm.load(f"models/{env_name}/{env_name}_{str(algorithm.__name__)}/model_mp", env=env,device="cuda")
    # for i in range(EVAL_EPISODES):
    #     done = truncated = False
    #     obs, info = env.reset()
    #     tot_rew = 0
    #     while not (done or truncated):
    #         action, _states = model.predict(obs, deterministic=True)
    #         obs, reward, done, truncated, info = env.step(action[()])
    #         tot_rew += reward
    #     print(tot_rew)
    # env.close()