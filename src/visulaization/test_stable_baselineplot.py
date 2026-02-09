import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common import results_plotter

# Hardcoded path to the DIRECTORY containing the monitor file(s)
LOG_DIR = r".\logs\01\PlanWaypointEnv-v2"
LOG_DIR = r".\logs\01\FreeFlightCREnv-v0"

def plot_custom_results(log_folder, title='FreeFlight SAC Learning Curve'):
    """
    Load results and plot with a smoothing window to generate nice curves.
    Uses stable_baselines3.common.results_plotter helper functions.
    """
    if not os.path.exists(log_folder):
        print(f"Error: Directory {log_folder} does not exist.")
        return

    print(f"Reading monitor logs from: {log_folder}")

    try:
        # 1. Load data
        # results_plotter.load_results returns a pandas DataFrame with all monitor data
        df = results_plotter.load_results(log_folder)
    except IndexError:
        print(f"No monitor files found in {log_folder}")
        return
    except Exception as e:
        print(f"Error loading results: {e}")
        return

    # 2. Decompose to X and Y (Timesteps vs Rewards)
    # Using X_TIMESTEPS as requested
    x, y = results_plotter.ts2xy(df, results_plotter.X_TIMESTEPS)
    #y_2 = df["waypoints_completed"].values
    
    if len(x) == 0:
        print("No data found in logs.")
        return

    # 3. Define smoothing
    # Dynamic window size or fixed
    window_size = 50
    # Adjust window if data is scarce
    if len(x) < window_size:
        window_size = max(1, len(x) // 2)

    print(f"Plotting {len(x)} episodes with smoothing window {window_size}...")

    # 4. Calculate smoothed curve using SB3's window_func
    # window_func(var_1, var_2, window, func) -> (smoothed_var_1, smoothed_var_2)
    # We apply np.mean to the rolling window
    try:
        x_smoothed, y_smoothed = results_plotter.window_func(x, y, window_size, np.mean)
        #x_smoothed_2, y_smoothed_2 = results_plotter.window_func(x, y_2, window_size, np.mean)
    except AttributeError:
        # Fallback if window_func is not in this version of SB3
        print("results_plotter.window_func not found, using manual rolling mean.")
        weights = np.repeat(1.0, window_size) / window_size
        y_smoothed = np.convolve(y, weights, 'valid')
        x_smoothed = x[window_size - 1:]

    # 5. Plotting
    plt.figure(figsize=(12, 6))
    
    # Plot raw data (transparent background)
    plt.plot(x, y, alpha=0.2, color='steelblue', linewidth=1, label='Raw Episode Reward')
    
    # Plot smoothed data (solid foreground)
    plt.plot(x_smoothed, y_smoothed, color='#b22222', linewidth=2, label=f'Moving Average (n={window_size})')

    plt.xlabel('Timesteps')
    plt.ylabel('Episode Reward')
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Save
    save_path = os.path.join(os.getcwd(), "training_results_custom.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Plot successfully saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    plot_custom_results(LOG_DIR)
