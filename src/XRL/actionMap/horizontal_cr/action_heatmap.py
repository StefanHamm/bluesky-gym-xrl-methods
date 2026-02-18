import gymnasium as gym
from stable_baselines3 import SAC,TD3,DDPG,PPO
import bluesky_gym
from bluesky_gym.wrappers.xrlMethods.state.action.horizontal_cr_env_action_heatmap import ActionHeatmapWrapper
bluesky_gym.register_envs()



if __name__ == "__main__":
    JOBID = "4675598"
    SEED = 42
    DEBUG = False
    # Initialize the environment and logger
    env_name = 'HorizontalCREnv-v0'

    if DEBUG:
        gifFolder = f"./plots/{JOBID}/{env_name}/actionHeatmapDebug/"
    else:
        gifFolder = f"./plots/{JOBID}/{env_name}/actionHeatmap/"

    env = gym.make(env_name,render_mode='human')
    env.reset(seed=SEED)
    

    modelpath = f"models/{JOBID}/HorizontalCREnv-v0/HorizontalCREnv-v0_SAC_singleEnv_baseline_model_mp.zip"
    model = SAC.load(modelpath,device='cpu')
    
    actionHeatmap = ActionHeatmapWrapper(env, model=model,draw_action_heatmap=True, grid_size=9, grid_spacing_km=5,export_gifs_path=gifFolder,fps=5,plot_action_path=True)
    
    episodes = 10
    for ep in range(episodes):
        done = truncated = False
        obs, info = actionHeatmap.reset()
        step = 0
        
        while not (done or truncated):
            step+=1
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = actionHeatmap.step(action)