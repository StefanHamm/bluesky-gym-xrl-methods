import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import os
import pandas as pd
import argparse
import logging
import json
import shutil
import tempfile
from stable_baselines3.common import results_plotter

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def load_csv(file_path, metrics, window_size=-1, smoothing_percentage=0.01, align='right'):
    try:
        algorithm_name = os.path.basename(file_path).split('_')[1]
    except IndexError:
        algorithm_name = os.path.basename(file_path).replace('.csv', '').replace('.monitor', '')
        
    logging.info(f"Loading data for algorithm: {algorithm_name}")
    
    df = pd.DataFrame()

    # Use results_plotter.load_results via a temporary directory to handle parsing
    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            # Copy file to temp dir so load_results can find it
            tmp_path = os.path.join(tmp_dir, os.path.basename(file_path))
            shutil.copy(file_path, tmp_path)
            
            # load_results returns a dataframe with all monitor data in the path
            df = results_plotter.load_results(tmp_dir)
            
        except Exception as e:
            logging.error(f"Error reading {file_path}: {e}")
            return pd.DataFrame()

    df['algorithm'] = algorithm_name
    
    # Handle Timesteps
    # Monitor files usually have 'l' (episode length) and 'r' (reward)
    if 'l' in df.columns:
        df['timesteps'] = df['l'].cumsum()
    elif 'timesteps' not in df.columns:
        # Fallback index as timesteps if nothing else
        df['timesteps'] = df.index
    
    # Calculate moving averages for requested metrics
    
    # Determine window size
    if window_size <= 0:
        # Auto: percentage of number of episodes
        adj_window_size = int(len(df) * smoothing_percentage)
        adj_window_size = max(1, adj_window_size)
    else:
        adj_window_size = window_size

    df['window_size'] = adj_window_size

    if len(df) < adj_window_size:
        # Adjust window if data is scarce but keep it at least 1
        adj_window_size = max(1, len(df) // 2)

    for metric in metrics:
        ma_col = f'{metric}_ma'
        if metric in df.columns:
            if len(df) >= adj_window_size:
                df[ma_col] = np.nan
                # ma_values = moving_average(df[metric].values, window_size)
                _, ma_values = results_plotter.window_func(df['timesteps'].values, df[metric].values, adj_window_size, np.mean)
                
                if align == 'right':
                    df.loc[adj_window_size-1:, ma_col] = ma_values
                elif align == 'left':
                    df.loc[:len(df)-adj_window_size, ma_col] = ma_values
            else:
                df[ma_col] = df[metric]
        else:
            logging.warning(f"Metric {metric} not found in {file_path}")
            df[ma_col] = np.nan
        
    return df

def load_folder_data(folder_path, metrics, window_size=-1, smoothing_percentage=0.01, align='right'):
    # Look for .monitor.csv or .csv files
    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv') or f.endswith('.monitor.csv')]
    # Also handle files that might just be .monitor if that's the naming convention, 
    # but standard SB3 is .monitor.csv usually.
    
    df_list = [load_csv(f, metrics, window_size=window_size, smoothing_percentage=smoothing_percentage, align=align) for f in all_files]
    # Filter out empty dataframes
    df_list = [df for df in df_list if not df.empty]
    
    if not df_list:
        return pd.DataFrame()
    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df

def plot_metric(pd_data, metric, title, save_path, smoothing_info=""):
    plt.style.use(['science', 'notebook', 'grid'])
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ma_col = f'{metric}_ma'
    
    for algorithm, group in pd_data.groupby('algorithm'):
        group = group.sort_values('timesteps')
        # Check if we have MA data, otherwise raw
        y_values = group[ma_col] if ma_col in group.columns and not group[ma_col].isna().all() else group[metric]
        valid_indices = ~y_values.isna()
        if valid_indices.any():
            ax.plot(group.loc[valid_indices, 'timesteps'], y_values[valid_indices], marker='', label=algorithm)

    # Dynamically set x-ticks based on rounded max timesteps
    max_timestep = pd_data['timesteps'].max()
    if max_timestep <= 0:
        xticks = [0]
        xticklabels = ['0']
    else:
        step = 1e6
        rounded_max = int(round(max_timestep / step)) * int(step)
        n_ticks = int(rounded_max // step)
        xticks = [int(step * i) for i in range(1, n_ticks + 1)]
        xticks = [0] + xticks
        xticklabels = [str(int(x)) if x == 0 else f'{int(x/1e6)}e6' for x in xticks]

    ax.set_title(title)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel(f'Timesteps {smoothing_info}')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.legend()

    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser(description="Generic metric plotter for BlueSky Gym Envs")
    parser.add_argument("--jobid", type=str, required=True, help="Job identifier")
    parser.add_argument("--env_name", type=str, required=True, help="Environment Name (e.g., SectorCREnv-v0)")
    parser.add_argument("--metrics", type=str, nargs='+', required=True, help="List of metrics to plot (columns in csv)")
    parser.add_argument("--align", type=str, choices=['left', 'right'], default='right', help="Alignment of moving average")
    parser.add_argument("--window_size", type=int, default=-1, help="Window size for moving average (Default -1 for Auto)")
    parser.add_argument("--smoothing_percentage", type=float, default=0.01, help="Percentage for auto window size (Default 0.01)")
    args = parser.parse_args()
    
    job_id = args.jobid
    env_name = args.env_name
    metrics = args.metrics
    
    logDir = f"./logs/{job_id}/{env_name}/"
    plotDir = f"./plots/{job_id}/"
    os.makedirs(plotDir, exist_ok=True)
    
    if os.path.exists(logDir):
        logging.info(f"Processing environment: {env_name} with metrics: {metrics}")
        combined_df = load_folder_data(logDir, metrics, window_size=args.window_size, smoothing_percentage=args.smoothing_percentage, align=args.align)
        
        if not combined_df.empty:
             # Determine smoothing info label for caption
            avg_ws = int(combined_df['window_size'].mean()) if 'window_size' in combined_df.columns else 0
            if args.window_size <= 0:
                 smoothing_label = f"(Smoothing: n={avg_ws} = {args.smoothing_percentage:.1%})"
            else:
                 smoothing_label = f"(Smoothing: n={avg_ws})"

            for metric in metrics:
                title = f'{env_name} - {metric}'
                save_path = os.path.join(plotDir, f"{env_name}_{metric}_{args.align}.pdf")
                logging.info(f"Plotting {metric} to {save_path}")
                try:
                    plot_metric(combined_df, metric, title, save_path, smoothing_info=smoothing_label)
                except Exception as e:
                    logging.error(f"Failed to plot {metric}: {e}")
        else:
            logging.warning(f"No valid data found for {env_name} in {logDir}")
    else:
        logging.error(f"Log directory for {env_name} not found at {logDir}")
