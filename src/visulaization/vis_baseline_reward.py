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

def plot_rewards(pd_reward, title, save_path, smoothing_info=""):
    plt.style.use(['science', 'notebook', 'grid'])
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Group by algorithm and plot each one separately
    for algorithm, group in pd_reward.groupby('algorithm'):
        group = group.sort_values('timesteps')
        # Plot moving average if available, otherwise raw reward
        y_values = group['total_reward_ma'] if 'total_reward_ma' in group.columns and not group['total_reward_ma'].isna().all() else group['total_reward']
        # Filter out NaNs for plotting
        valid_indices = ~y_values.isna()
        ax.plot(group.loc[valid_indices, 'timesteps'], y_values[valid_indices], marker='', label=algorithm) # Removed marker='o' for cleaner line plot

    ax.set_title(title)
    ax.set_xticks([0,1e6,2e6])
    ax.set_xticklabels(['0','1e6','2e6'])
    ax.set_xlabel(f'Timesteps {smoothing_info}')
    ax.set_ylabel('Reward')
    ax.legend()
    plt.savefig(save_path, format='pdf',bbox_inches='tight')
    plt.close(fig) # Close the figure to free memory

def plot_multipanel(env_data_dict, save_path, window_size_arg=-1, smoothing_percentage_arg=0.01):
    plt.style.use(['science', 'notebook', 'grid'])
    
    num_envs = len(env_data_dict)
    cols = 2
    rows = (num_envs + 1) // cols
    
    fig, axs = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axs = axs.flatten()
    
    for i, (env_name, df) in enumerate(env_data_dict.items()):
        ax = axs[i]
        for algorithm, group in df.groupby('algorithm'):
            group = group.sort_values('timesteps')
            y_values = group['total_reward_ma'] if 'total_reward_ma' in group.columns and not group['total_reward_ma'].isna().all() else group['total_reward']
            valid_indices = ~y_values.isna()
            ax.plot(group.loc[valid_indices, 'timesteps'], y_values[valid_indices], marker='', label=algorithm)
        
        # Calculate smoothing info label
        avg_ws = int(df['window_size'].mean()) if 'window_size' in df.columns else 0
        if window_size_arg <= 0:
             local_smoothing_info = f"(Smoothing: n={avg_ws} = {smoothing_percentage_arg:.1%})"
        else:
             local_smoothing_info = f"(Smoothing: n={avg_ws})"

        ax.set_title(env_name)
        ax.set_xlabel(f'Timesteps {local_smoothing_info}')
        ax.set_ylabel('Reward')
        ax.set_xticks([0, 1e6, 2e6])
        ax.set_xticklabels(['0', '1e6', '2e6'])
        
    # Hide unused subplots
    for j in range(i + 1, len(axs)):
        axs[j].axis('off')
        
    # Add a single legend to the last subplot (or outside)
    handles, labels = axs[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='lower center', ncol=len(handles), bbox_to_anchor=(0.5, 0.01))
        
    plt.tight_layout(rect=[0, 0.05, 1, 1]) # Adjust layout to make room for legend
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close(fig)

def load_csv(file_path, window_size=-1, smoothing_percentage=0.01, align='right'):
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
            
    # Check if this is a Monitor file (has 'r' for reward and 'l' for length)
    if 'r' in df.columns and 'l' in df.columns:
        x, y = results_plotter.ts2xy(df, results_plotter.X_TIMESTEPS)
        df['timesteps'] = x
        df['total_reward'] = y
    else:
        # Fallback or pass through if already formatted
        if 'timesteps' not in df.columns and 'l' in df.columns:
            df['timesteps'] = df['l'].cumsum()
        if 'total_reward' not in df.columns and 'r' in df.columns:
            df['total_reward'] = df['r']
            
    # add a column with the algorithm name
    df['algorithm'] = algorithm_name
    
    # Calculate moving average for total_reward
    total_reward_values = df['total_reward'].values
    timesteps_values = df['timesteps'].values

    # Determine window size
    if window_size <= 0:
        # Auto: percentage of number of episodes
        adj_window_size = int(len(df) * smoothing_percentage)
        adj_window_size = max(1, adj_window_size)
    else:
        adj_window_size = window_size
    
    # Store the window size in the dataframe if we want to retrieve it later, 
    # though for plotting aggregated lines it is per line.
    
    df['window_size'] = adj_window_size

    if len(df) < adj_window_size:
        # Adjust window if data is scarce but keep it at least 1
        adj_window_size = max(1, len(df) // 2)
        
    df['total_reward_ma'] = np.nan
    
    if len(df) >= adj_window_size:
        # Use SB3 window_func logic
        _, ma_values = results_plotter.window_func(timesteps_values, total_reward_values, adj_window_size, np.mean)
        
        if align == 'right':
            df.loc[adj_window_size-1:, 'total_reward_ma'] = ma_values
        elif align == 'left':
            df.loc[:len(df)-adj_window_size, 'total_reward_ma'] = ma_values
    else:
        df['total_reward_ma'] = df['total_reward'] # Fallback if not enough data
        
    return df

def load_folder_data(folder_path, window_size=-1, smoothing_percentage=0.01, align='right'):
    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
    df_list = [load_csv(f, window_size=window_size, smoothing_percentage=smoothing_percentage, align=align) for f in all_files]
    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # add argument for the jobID to load the correct folder
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobid", type=str, default=None, help="Job identifier")
    parser.add_argument("--multipanel", action="store_true", help="Plot all environments in a single multipanel figure")
    parser.add_argument("--align", type=str, choices=['left', 'right'], default='right', help="Alignment of moving average (left or right)")
    parser.add_argument("--window_size", type=int, default=-1, help="Window size for moving average (Default -1 for Auto)")
    parser.add_argument("--smoothing_percentage", type=float, default=0.01, help="Percentage for auto window size (Default 0.01)")
    args = parser.parse_args()
    
    logDir = f"./logs/{args.jobid}/"
    #create a folder in ./plots/ with the jobid
    plotDir = f"./plots/{args.jobid}/"
    os.makedirs(plotDir, exist_ok=True)
    
    # get all folders in logDir
    env_folders = [f.path for f in os.scandir(logDir) if f.is_dir()]
    logging.info(f"Found environment folders: {env_folders}")
    
    env_data_dict = {}

    # load data for each environment
    for env_folder in env_folders:
        env_name = os.path.basename(env_folder)
        logging.info(f"Processing environment: {env_name}")
        combined_df = load_folder_data(env_folder, window_size=args.window_size, smoothing_percentage=args.smoothing_percentage, align=args.align)
        
        if args.multipanel:
            env_data_dict[env_name] = combined_df
        else:
            # Determine smoothing info label for caption
            avg_ws = int(combined_df['window_size'].mean()) if 'window_size' in combined_df.columns else 0
            if args.window_size <= 0:
                 smoothing_label = f"(Smoothing: n={avg_ws} = {args.smoothing_percentage:.1%})"
            else:
                 smoothing_label = f"(Smoothing: n={avg_ws})"

            export_path = os.path.join(plotDir, f"{env_name}_combined_rewards.csv")
            plot_rewards(combined_df, f'Reward Comparison for {env_name}', os.path.join(plotDir, f"{env_name}_reward_comparison_{args.align}.pdf"), smoothing_info=smoothing_label)

    if args.multipanel and env_data_dict:
        logging.info("Generating multipanel plot...")
        plot_multipanel(env_data_dict, os.path.join(plotDir, f"multipanel_reward_comparison_{args.align}.pdf"), window_size_arg=args.window_size, smoothing_percentage_arg=args.smoothing_percentage)