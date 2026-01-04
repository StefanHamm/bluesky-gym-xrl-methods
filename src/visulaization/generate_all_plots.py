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
    args = parser.parse_args()
    
    job_id = args.jobid
    align = "right"
    
    # 1. Baseline Reward - Single Plots
    run_script("vis_baseline_reward.py", ["--jobid", job_id, "--align", align])
    
    # 2. Baseline Reward - Multipanel
    run_script("vis_baseline_reward.py", ["--jobid", job_id, "--multipanel", "--align", align])
    
    # 3. Sector Intrusions
    run_script("vis_sector_intrusions.py", ["--jobid", job_id, "--align", align])
    
    # 4. Horizontal Intrusions
    run_script("vis_horizontal_intrusions.py", ["--jobid", job_id, "--align", align])
    
    # 5. Static Obstacle Metrics
    run_script("vis_static_obstacle_metrics.py", ["--jobid", job_id, "--align", align])
    
    logging.info("All visualization scripts completed.")
