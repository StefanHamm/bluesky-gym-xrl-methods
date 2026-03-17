import gymnasium as gym
from stable_baselines3 import SAC,TD3,DDPG,PPO
import bluesky as bs
import bluesky_gym
import traceback
#from bluesky_gym.wrappers.xrlMethods.state.action.sector_cr_env_action_heatmap import ActionHeatmapWrapper
print(bs.sim)
original_init = bs.init
def traced_init(*args, **kwargs):
    print("---------------")
    print("bs.init called with:", args, kwargs)
    traceback.print_stack()
    return original_init(*args, **kwargs)
bs.init = traced_init
bluesky_gym.register_envs()
env = gym.make("PlanWaypointEnv")
import bluesky as bs
print(bs.__file__)

if __name__ == "__main__":
    JOBID = "5124851"
    SEED = 43
    DEBUG = False
    # Initialize the environment and logger
    env_name = 'NavWaypointEvadeEnv-v0'
    #env_name ="PlanWaypointEnv"
    #spawnFactor = 2
  

    if DEBUG:
        gifFolder = f"./plots/{JOBID}/{env_name}/actionHeatmap/"
    else:
        gifFolder = f"./plots/{JOBID}/{env_name}/actionHeatmap/"

    env = gym.make(env_name,render_mode='human',window_width=1000,window_height=1000,plot_all_points = True,stencil_radius_in_km=400)
    env.reset(seed=SEED)
    
    modelpath = rf"models\{JOBID}\NavWaypointEvadeEnv-v0\NavWaypointEvadeEnv-v0_SAC_vecEnvLogs_baseline_model_mp.zip"
    model = SAC.load(modelpath,device='cpu')
    
    #actionHeatmap = ActionHeatmapWrapper(env, model=model,draw_action_heatmap=True, grid_size=9, grid_spacing_km=5,export_gifs_path=gifFolder,plot_action_path=True)
    

    episodes = 10
    for ep in range(episodes):
        done = truncated = False
        #obs, info = actionHeatmap.reset()
        obs,info = env.reset()
        step = 0

        while not (done or truncated):
            step+=1
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)