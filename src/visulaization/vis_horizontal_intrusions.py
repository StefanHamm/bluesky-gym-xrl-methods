import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import os
import pandas as pd
import argparse
import logging

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def load_csv(file_path, metric, align='right'):
    algorithm_name = os.path.basename(file_path).split('_')[1]
    logging.info(f"Loading data for algorithm: {algorithm_name}")
    df = pd.read_csv(file_path)
    df['algorithm'] = algorithm_name
    
    window_size = 1000
    ma_col = f'{metric}_ma'
    
    if metric in df.columns:
        if len(df) >= window_size:
            df[ma_col] = np.nan
            ma_values = moving_average(df[metric].values, window_size)
            if align == 'right':
                df.loc[window_size-1:, ma_col] = ma_values
            elif align == 'left':
                df.loc[:len(df)-window_size, ma_col] = ma_values
        else:
            df[ma_col] = df[metric]
    else:
        logging.warning(f"Metric {metric} not found in {file_path}")
        df[ma_col] = np.nan
        
    return df

def load_folder_data(folder_path, metric, align='right'):
    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
    df_list = [load_csv(f, metric, align=align) for f in all_files]
    if not df_list:
        return pd.DataFrame()
    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df

def plot_metric(pd_data, metric, title, save_path):
    plt.style.use(['science', 'notebook', 'grid'])
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ma_col = f'{metric}_ma'
    
    for algorithm, group in pd_data.groupby('algorithm'):
        group = group.sort_values('timesteps')
        y_values = group[ma_col] if ma_col in group.columns and not group[ma_col].isna().all() else group[metric]
        valid_indices = ~y_values.isna()
        ax.plot(group.loc[valid_indices, 'timesteps'], y_values[valid_indices], marker='', label=algorithm)

    ax.set_title(title)
    ax.set_xticks([0,1e6,2e6])
    ax.set_xticklabels(['0','1e6','2e6'])
    ax.set_xlabel('Timesteps')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.legend()
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobid", type=str, required=True, help="Job identifier")
    parser.add_argument("--align", type=str, choices=['left', 'right'], default='right', help="Alignment of moving average")
    args = parser.parse_args()
    
    env_name = "HorizontalCREnv-v0"
    metric = "total_intrusions"
    
    logDir = f"./logs/{args.jobid}/{env_name}/"
    plotDir = f"./plots/{args.jobid}/"
    os.makedirs(plotDir, exist_ok=True)
    
    if os.path.exists(logDir):
        logging.info(f"Processing environment: {env_name}")
        combined_df = load_folder_data(logDir, metric, align=args.align)
        if not combined_df.empty:
            plot_metric(combined_df, metric, f'{env_name} - {metric}', os.path.join(plotDir, f"{env_name}_{metric}_{args.align}.pdf"))
        else:
            logging.warning(f"No data found for {env_name}")
    else:
        logging.error(f"Log directory for {env_name} not found at {logDir}")
