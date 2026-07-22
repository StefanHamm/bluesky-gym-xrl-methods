import argparse
import subprocess
import sys
import os
import logging

def run_script(script_name, args):
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_dir, script_name)
    
    cmd = [sys.executable, script_path] + args
    logging.info(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logging.info(f"Output of {script_name}:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running {script_name}:\n{e.stderr}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser(description="Generate all plots for a specific job ID.")
    parser.add_argument("--jobid", type=str, required=True, help="Job identifier")
    parser.add_argument("--window_size", type=int, default=-1, help="Window size for moving average smoothing. Default -1 (Auto: 1% of episodes)")
    parser.add_argument("--smoothing_percentage", type=float, default=0.01, help="Percentage of episodes to use for auto window size (Default: 0.01)")
    args = parser.parse_args()
    
    job_id = args.jobid
    align = "right"
    window_size = str(args.window_size)
    smoothing_percentage = str(args.smoothing_percentage)
    
    # 1. Baseline Reward - Single Plots
    run_script("vis_baseline_reward.py", ["--jobid", job_id, "--align", align, "--window_size", window_size, "--smoothing_percentage", smoothing_percentage])
    
    # 2. Baseline Reward - Multipanel
    run_script("vis_baseline_reward.py", ["--jobid", job_id, "--multipanel", "--align", align, "--window_size", window_size, "--smoothing_percentage", smoothing_percentage])
    
    # 3. Generic Plots for Extra Metrics
    # Define mapping of environment to extra keywords (metrics)
    keywords_mapping = {
        "FreeFlightCREnv-v0": ['total_intrusions'],
        "PlanWaypointEnv-v2": ['waypoints_completed'],
        "PlanWaypointEnv-v0": ['waypoints_completed'],
        "PlanWaypointEvadeEnv-v0": ['waypoints_completed', 'total_intrusions'],
        "HorizontalCREnv-v0": ['total_intrusions', 'average_drift'],
        "SectorCREnv-v0": ['total_intrusions', 'average_drift'],
        "StaticObstacleEnv-v0": ['crashed', 'average_drift', 'waypoint_reached'],
        "NavWaypointEvadeEnv-v0" : ["drift_mean","corridor_leave_mean","intrusion_count","obstacle_intrusion_count","waypoint_reached_count","path_length","crash"]
    }

    logDir = f"./logs/{job_id}/"
    if os.path.exists(logDir):
        # Scan for environment folders
        found_envs = [f.name for f in os.scandir(logDir) if f.is_dir()]
        logging.info(f"Found environments in logs: {found_envs}")
        
        for env_name in found_envs:
            if env_name in keywords_mapping:
                metrics = keywords_mapping[env_name]
                logging.info(f"Generating extra plots for {env_name}: {metrics}")
                # Pass metrics as separate arguments
                run_script("vis_generic_metric.py", ["--jobid", job_id, "--env_name", env_name, "--metrics"] + metrics + ["--align", align, "--window_size", window_size, "--smoothing_percentage", smoothing_percentage])
            else:
                logging.info(f"No extra keyword mapping found for {env_name}, skipping generic plots.")
    else:
        logging.error(f"Log directory {logDir} does not exist.")
    
    logging.info("All visualization scripts completed.")
