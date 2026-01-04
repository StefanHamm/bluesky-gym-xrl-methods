import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import os
import pandas as pd
import argparse
import logging


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def plot_rewards(pd_reward, title, save_path):
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
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Reward')
    ax.legend()
    plt.savefig(save_path, format='svg',bbox_inches='tight')
    plt.close(fig) # Close the figure to free memory

def plot_multipanel(env_data_dict, save_path):
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
        
        ax.set_title(env_name)
        ax.set_xlabel('Timesteps')
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
    plt.savefig(save_path, format='svg', bbox_inches='tight')
    plt.close(fig)

def load_csv(file_path, align='right'):
    algorithm_name = os.path.basename(file_path).split('_')[1]
    logging.info(f"Loading data for algorithm: {algorithm_name}")
    df = pd.read_csv(file_path)
    # add a column with the algorithm name
    df['algorithm'] = algorithm_name
    
    # Calculate moving average for total_reward
    window_size = 1000  # Adjust window size as needed
    if len(df) >= window_size:
        df['total_reward_ma'] = np.nan
        ma_values = moving_average(df['total_reward'].values, window_size)
        if align == 'right':
            df.loc[window_size-1:, 'total_reward_ma'] = ma_values
        elif align == 'left':
            df.loc[:len(df)-window_size, 'total_reward_ma'] = ma_values
    else:
        df['total_reward_ma'] = df['total_reward'] # Fallback if not enough data
        
    return df

def load_folder_data(folder_path, align='right'):
    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
    df_list = [load_csv(f, align=align) for f in all_files]
    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # add argument for the jobID to load the correct folder
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobid", type=str, default=None, help="Job identifier")
    parser.add_argument("--multipanel", action="store_true", help="Plot all environments in a single multipanel figure")
    parser.add_argument("--align", type=str, choices=['left', 'right'], default='right', help="Alignment of moving average (left or right)")
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
        combined_df = load_folder_data(env_folder, align=args.align)
        
        if args.multipanel:
            env_data_dict[env_name] = combined_df
        else:
            export_path = os.path.join(plotDir, f"{env_name}_combined_rewards.csv")
            plot_rewards(combined_df, f'Reward Comparison for {env_name}', os.path.join(plotDir, f"{env_name}_reward_comparison_{args.align}.svg"))

    if args.multipanel and env_data_dict:
        logging.info("Generating multipanel plot...")
        plot_multipanel(env_data_dict, os.path.join(plotDir, f"multipanel_reward_comparison_{args.align}.svg"))

