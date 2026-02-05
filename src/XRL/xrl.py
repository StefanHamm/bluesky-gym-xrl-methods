import gymnasium as gym
from stable_baselines3 import SAC, TD3, DDPG, PPO
import bluesky_gym
import numpy as np
import logging
import argparse
import os
import sys

# Import Wrappers
from bluesky_gym.wrappers.xrlMethods.state.saliency.horizontal_cr_env_saliency import SaliencyHorizontalControl
from bluesky_gym.wrappers.xrlMethods.state.saliency.sector_cr_env_saliency import SaliencySectorControl
from bluesky_gym.wrappers.xrlMethods.state.saliency.plan_waypoint_env_saliency import SaliencyPlanWaypoint
from bluesky_gym.wrappers.xrlMethods.state.action.horizontal_cr_env_action_heatmap import ActionHeatmapWrapper
from bluesky_gym.wrappers.xrlMethods.state.action.sector_cr_env_action_heatmap import ActionHeatmapWrapper as SectorActionHeatmapWrapper
from bluesky_gym.wrappers.xrlMethods.state.action.plan_waypoint_env_action_heatmap import ActionHeatmapWrapper as PlanWaypointActionHeatmapWrapper
# Import Explainers
# Assuming src is in python path, e.g. run via python -m src.XRL.xrl or from root
# We need to ensure import works.
try:
    from src.XRL.shapMethods.shap_explainers import runSafeStateExplainer, runBackgroundExplainer
except ImportError:
    # Try local import if running from same folder or adjust path
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))
    from src.XRL.shapMethods.shap_explainers import runSafeStateExplainer, runBackgroundExplainer


bluesky_gym.register_envs()

# --- Configs ---

def get_safe_vals(env_name):
    if env_name == 'HorizontalCREnv-v0':
        return {
            "intruder_distance": 0.5,
            "cos_difference_pos": -1.0,  # Behind
            "sin_difference_pos": 0.0,
            "x_difference_speed": 1.0,    # Flying away
            "y_difference_speed": 1.0
        }
    elif env_name == 'PlanWaypointEnv-v0':
         return {
            "waypoint_reached": 1.0 
        }
    elif env_name == 'SectorCREnv-v0':
        D_NORTH = 0
        D_EAST = 30000 
        vx_r_norm = -0.5
        vy_r_norm = 0.0
        vx_r_raw = vx_r_norm * 32
        vy_r_raw = vy_r_norm * 66
        track_rad = np.arctan2(vy_r_raw, vx_r_raw)
        return {
            "x_r": D_NORTH/13000,
            "y_r": D_EAST/13000,
            "vx_r": vx_r_norm,
            "vy_r": vy_r_norm,
            "cos(track)": np.cos(track_rad),
            "sin(track)": np.sin(track_rad),
            "distances": (np.sqrt(D_NORTH**2 + D_EAST**2)-50000)/15000
        }
    return {}

def get_feature_mapping(env_name):
    if env_name == 'HorizontalCREnv-v0':
        return [
        "intruder_distance","cos_difference_pos","sin_difference_pos",
        "x_difference_speed","y_difference_speed"]
    elif env_name == 'SectorCREnv-v0':
        return ["x_r","y_r","vx_r","vy_r",
        "cos(track)","sin(track)","distances"]
    return []

def get_background_data_path(env_name):
    base = os.path.dirname(__file__)
    # Relative to src/XRL/xrl.py
    if env_name == 'HorizontalCREnv-v0':
        return os.path.join(base, "shapMethods", "horizontal_cr", "calculate_shap_background_data", "intruder_background.npy")
    elif env_name == 'SectorCREnv-v0':
        return os.path.join(base, "shapMethods", "sector_cr", "calculate_shap_background_data", "intruder_background.npy")
    return None

def main():
    parser = argparse.ArgumentParser(description='XRL Method Runner')
    
    # Core args
    parser.add_argument('--method', type=str, required=True, choices=['shap_safe_state', 'shap_background', 'action_heatmap'], help='XRL Method to run')
    parser.add_argument('--env', type=str, required=True, help='Environment Name (e.g., HorizontalCREnv-v0)')
    parser.add_argument('--jobid', type=str, required=True, help='Job ID for model loading')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Run params
    parser.add_argument('--n_eps', type=int, default=10, help='Number of episodes')
    parser.add_argument('--max_steps', type=int, default=1000, help='Max steps per episode (default high, early stop usually)')
    
    # Model params
    parser.add_argument('--algo', type=str, default=None, choices=['SAC', 'TD3', 'DDPG', 'PPO','A2C'], help='RL Algorithm. If not provided, tries to guess based on env defaults.')
    
    # Visualization params
    parser.add_argument('--color_mode', type=str, default="default", help='Color mode for SHAP: default, clipped, or scaled')
    parser.add_argument('--plot_action_path', action='store_true', default=True, help='Plot action path (SAFE STATE only)')
    parser.add_argument('--plot_safe_path', action='store_true', help='Plot safe path (SAFE STATE only)')
    parser.add_argument('--debug', action='store_true', help='Debug mode (logging and paths)')
    
    # Heatmap params
    parser.add_argument('--grid_size', type=int, default=10, help='Grid size for heatmap')
    parser.add_argument('--grid_spacing', type=int, default=5, help='Grid spacing km for heatmap')

    # Env specific
    parser.add_argument('--spawn_factor', type=int, default=None, help='Spawn Factor (SectorCR)')
    parser.add_argument('--point_to_waypoint', action='store_true', help='Point heatmap to waypoint (HorizontalCR only)')

    args = parser.parse_args()

    # Defaults
    if args.algo is None:
        if 'Horizontal' in args.env or 'PlanWaypoint' in args.env:
            args.algo = 'SAC'
        elif 'Sector' in args.env:
            args.algo = 'TD3'
        else:
            raise ValueError("Algorithm must be specified for unknown env.")

    # Logging
    log_level = logging.DEBUG if args.debug else logging.WARNING
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

    # Paths
    # plots/{JOBID}/{env_name}/{method}/{color_mode}/
    
    method_dir = ""
    if args.method == 'shap_safe_state':
        if args.debug:
            method_dir = "shapSafeStateDebug"
        elif args.plot_action_path:
             method_dir = f"shapSafeState/{args.color_mode}"
        else:
            method_dir = f"shapSafeState/{args.color_mode}"
    elif args.method == 'shap_background':
        method_dir = "shapBackgroundDataDebug" if args.debug else f"shapBackgroundData/{args.color_mode}"
    elif args.method == 'action_heatmap':
        method_dir = "actionHeatmapDebug" if args.debug else "actionHeatmap"
         
        if args.env == 'HorizontalCREnv-v0' and args.point_to_waypoint:
            method_dir += "/PointToWaypoint"
        else:
            method_dir += "/KeepHeading"

    gif_folder = f"./plots/{args.jobid}/{args.env}/{method_dir}/"
    
    # Model Path
    model_path = f"models/{args.jobid}/{args.env}/{args.env}_{args.algo}_singleEnv_baseline_model_mp.zip"
    
    # Environment Options
    env_options = {}
    if args.spawn_factor is not None:
        env_options['SpawnFactor'] = args.spawn_factor
    elif args.env == 'SectorCREnv-v0':
         env_options['SpawnFactor'] = 2 # Default in scripts
         
    
    # Make Env
    print(f"Making environment: {args.env}")
    env = gym.make(args.env, render_mode="human")
    env.metadata['render_fps'] = 800
    print(f"Resetting environment with seed: {args.seed}")
    env.reset(seed=args.seed, options=env_options)

    wrapper = None
    
    # Load Model Class
    ModelClass = getattr(sys.modules[__name__], args.algo) 
    
    print(f"Running method: {args.method}")

    # Wrapper Initialization
    if args.method == 'shap_safe_state':
        # Load model first
        print(f"Loading model from: {model_path}")
        model = ModelClass.load(model_path, device='cpu')
        
        safe_vals = get_safe_vals(args.env)
        
        if args.env == 'HorizontalCREnv-v0':
            wrapper = SaliencyHorizontalControl(env, safe_vals, args.debug, export_gifs_path=gif_folder, fps=5, color_mode=args.color_mode, plot_action_path=args.plot_action_path, plot_safe_path=args.plot_safe_path, model=model)
        elif args.env == 'SectorCREnv-v0':
            wrapper = SaliencySectorControl(env, safe_vals, args.debug, export_gifs_path=gif_folder, fps=5, color_mode=args.color_mode, plot_action_path=args.plot_action_path, plot_safe_path=args.plot_safe_path, model=model)
        elif args.env == 'PlanWaypointEnv-v0':
             # Note: SaliencyPlanWaypoint might not support plot_safe_path, check definition in other files if possible, or leave out if not sure.
             # Based on previous file, it was NOT passed.
             wrapper = SaliencyPlanWaypoint(env, safe_vals, args.debug, export_gifs_path=gif_folder, fps=5, color_mode=args.color_mode, plot_action_path=args.plot_action_path, model=model)
        
    elif args.method == 'shap_background':
        # Load model with env
        print(f"Loading model with env context from: {model_path}")
        model = ModelClass.load(model_path, device='cpu')
        # Create Empty Wrapper first
        if args.env == 'HorizontalCREnv-v0':
             wrapper = SaliencyHorizontalControl(env, None, None, export_gifs_path=gif_folder, fps=5,color_mode=args.color_mode,plot_action_path=args.plot_action_path,model=model)
        elif args.env == 'SectorCREnv-v0':
             wrapper = SaliencySectorControl(env, None, None, export_gifs_path=gif_folder, fps=5,color_mode=args.color_mode,plot_action_path=args.plot_action_path,model=model)
        
        
        
    elif args.method == 'action_heatmap':
        # Load model first
        print(f"Loading model from: {model_path}")
        model = ModelClass.load(model_path, device='cpu')
        
        if args.env == 'HorizontalCREnv-v0':
             wrapper = ActionHeatmapWrapper(env, model=model, draw_action_heatmap=True, grid_size=args.grid_size, grid_spacing_km=args.grid_spacing, export_gifs_path=gif_folder, fps=5, point_to_waypoint=args.point_to_waypoint, plot_action_path=args.plot_action_path)
        elif args.env == 'SectorCREnv-v0':
             wrapper = SectorActionHeatmapWrapper(env, model=model, draw_action_heatmap=True, grid_size=args.grid_size, grid_spacing_km=args.grid_spacing, export_gifs_path=gif_folder, fps=5, plot_action_path=args.plot_action_path)
        elif args.env == 'PlanWaypointEnv-v0':
             wrapper = PlanWaypointActionHeatmapWrapper(env, model=model, draw_action_heatmap=True, grid_size=args.grid_size, grid_spacing_km=args.grid_spacing, export_gifs_path=gif_folder, fps=5, plot_action_path=args.plot_action_path)

    if wrapper is None:
        print(f"Error: Could not initialize wrapper for env {args.env} and method {args.method}")
        return

    # Run Loop
    for i in range(args.n_eps):
        done = truncated = False
        obs, info = wrapper.reset(options=env_options)
        step = 0
        print(f"Starting Episode {i+1}...")
        
        while not (done or truncated) and step < args.max_steps:
            step += 1
            action, _states = model.predict(obs, deterministic=True)
            
            if args.method == 'shap_safe_state':
                safe_vals = get_safe_vals(args.env)
                baseline_action = [0.0] if args.env == 'PlanWaypointEnv-v0' else None
                shap_values = runSafeStateExplainer(model, obs, safe_vals, baseline_action)
                obs, reward, done, truncated, info = wrapper.step(action[()], shap_values)
                
            elif args.method == 'shap_background':
                feature_mapping = get_feature_mapping(args.env)
                bg_path = get_background_data_path(args.env)
                background_data = np.load(bg_path)
                shap_values = runBackgroundExplainer(model, obs, background_data, feature_mapping, n_samples=300)
                obs, reward, done, truncated, info = wrapper.step(action[()], shap_values)
                
            elif args.method == 'action_heatmap':
                obs, reward, done, truncated, info = wrapper.step(action)
        
        if step == args.max_steps:
             print("Max steps reached.")

        if hasattr(args.env, 'SectorCREnv-v0'):
            print("Exporting GIF...")
            wrapper.export_episode_gif()

    env.close()
    print("Done.")

if __name__ == "__main__":
    main()
