"""
Script to correlate SHAP values from Safe State Explainer and Background Data Explainer.
Compares the SHAP values assigned to each intruder for a number of episodes.
"""

import gymnasium as gym
from stable_baselines3 import SAC
import bluesky_gym
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import sys

# Ensure we can import from src if running from root or elsewhere
sys.path.append(os.getcwd())

try:
    from src.XRL.shapMethods.shap_explainers import runBackgroundExplainer, runSafeStateExplainer
except ImportError:
    # If standard import fails, try appending the src parent directory
    print("Could not import shap_explainers directly. Checking path...")
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")))
    from src.XRL.shapMethods.shap_explainers import runBackgroundExplainer, runSafeStateExplainer

bluesky_gym.register_envs()

def main():
    # Configuration
    env_name = 'HorizontalCREnv-v0'
    JOBID = "4675598"
    SEED = 42
    N_EPS = 100  # Number of episodes to run
    N_BG_SAMPLES = 300 # Reduced from 300 for speed, increase for better accuracy
    OUTPUT_FOLDER = f"plots/{JOBID}/shap_correlation_results"
    
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Paths
    # Attempt to locate the background data file
    possible_paths = [
        "src/XRL/shapMethods/horizontal_cr/calculate_shap_background_data/intruder_background.npy",
        "calculate_shap_background_data/intruder_background.npy",
        os.path.join(os.path.dirname(__file__), "calculate_shap_background_data/intruder_background.npy")
    ]
    
    backgroundDataPath = None
    for p in possible_paths:
        if os.path.exists(p):
            backgroundDataPath = p
            break
            
    if backgroundDataPath is None:
        print(f"Error: Could not find intruder_background.npy in {possible_paths}")
        return

    print(f"Loading background data from: {backgroundDataPath}")
    backgroundData = np.load(backgroundDataPath)
    
    modelpath = f"models/{JOBID}/HorizontalCREnv-v0/HorizontalCREnv-v0_SAC_singleEnv_baseline_model_mp.zip"
    if not os.path.exists(modelpath):
        print(f"Error: Could not find model at {modelpath}")
        return
        
    print("Initializing Environment and Model...")
    env = gym.make(env_name)
    env.reset(seed=SEED)
    
    model = SAC.load(modelpath, env=env, device='cpu')
    
    # Feature configurations
    intruder_feature_mapping = [
        "intruder_distance", "cos_difference_pos", "sin_difference_pos",
        "x_difference_speed", "y_difference_speed"
    ]
        
    SAFE_VALS = {
        "intruder_distance": 0.5,
        "cos_difference_pos": -1.0,  # Behind
        "sin_difference_pos": 0.0,
        "x_difference_speed": 1.0,    # Flying away
        "y_difference_speed": 1.0
    }
    
    all_bg_shap = []
    all_safe_shap = []
    all_bg_base = []
    all_safe_base = []
    
    print(f"Starting data collection for {N_EPS} episodes...")
    
    total_steps = 0
    
    for i in range(N_EPS):
        obs, info = env.reset()
        done = truncated = False
        episode_step = 0
        
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            
            # Calculates SHAP values
            # background explainer
            shap_bg = runBackgroundExplainer(model, obs, backgroundData, intruder_feature_mapping, n_samples=N_BG_SAMPLES)
            
            # safe state explainer
            shap_safe = runSafeStateExplainer(model, obs, SAFE_VALS)
            
            # Extract values: shape (n_intruders, n_actions)
            # The shap_values index 0 corresponds to the single sample we passed (the observation)
            vals_bg = shap_bg.values[0]
            vals_safe = shap_safe.values[0]

            # Extract base values (expected value or reference value)
            # shape (n_actions,)
            base_bg = shap_bg.base_values[0]
            base_safe = shap_safe.base_values[0]
            
            # We must ensure they have the same shape
            if vals_bg.shape != vals_safe.shape:
                print(f"Shape mismatch at ep {i} step {episode_step}: BG {vals_bg.shape} vs Safe {vals_safe.shape}")
                continue
                
            all_bg_shap.append(vals_bg)
            all_safe_shap.append(vals_safe)
            all_bg_base.append(base_bg)
            all_safe_base.append(base_safe)
            
            obs, reward, done, truncated, info = env.step(action)
            
            total_steps += 1
            episode_step += 1
            
            if total_steps % 50 == 0:
                print(f"  Processed {total_steps} steps total...")
        
        print(f"Episode {i+1}/{N_EPS} finished.")

    env.close()
    print("Data collection complete.")
    
    if not all_bg_shap:
        print("No SHAP values collected.")
        return

    # Process Results
    # Stack all arrays: (Total_Samples * n_intruders, n_outputs)
    # Each element in all_bg_shap is shape (n_intruders_in_step, n_outputs)
    # We want to flatten the list of arrays into one big array
    
    # Check if empty (e.g. if episodes were shorter than expected or failed)
    if not all_bg_shap:
        print("No data collected in all_bg_shap")
        return

    try:
        bg_array = np.vstack(all_bg_shap)     # Shape: (N_total_intruder_samples, n_outputs)
        safe_array = np.vstack(all_safe_shap) # Shape: (N_total_intruder_samples, n_outputs)
    except ValueError as e:
        print(f"Error vstacking arrays: {e}")
        return
    
    print(f"Collected {bg_array.shape[0]} intruder-samples.")
    
    n_outputs = bg_array.shape[1]
    
    def create_correlation_plot(bg_arr, safe_arr, filename_suffix, title_suffix, threshold=0.0):
        plt.figure(figsize=(10, 8))
        colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
        
        reg_x = []
        reg_y = []
        has_data = False
        
        print(f"--- Generating Plot: {filename_suffix} (Threshold: {threshold}) ---")

        for dim in range(n_outputs):
            v1 = bg_arr[:, dim]
            v2 = safe_arr[:, dim]
            
            # Remove NaNs
            valid_mask = ~np.isnan(v1) & ~np.isnan(v2)
            v1 = v1[valid_mask]
            v2 = v2[valid_mask]
            
            # Filter
            if threshold > 0:
                # Keep point if either method assigns absolute value >= threshold
                mask = (np.abs(v1) >= threshold) | (np.abs(v2) >= threshold)
                v1 = v1[mask]
                v2 = v2[mask]
            
            if len(v1) < 2:
                print(f"Not enough data for dimension {dim}")
                continue
                
            has_data = True
            reg_x.append(v1)
            reg_y.append(v2)
            
            corr, p_value = pearsonr(v1, v2)
            print(f"Dim {dim}: r={corr:.4f} (p={p_value:.4g})")
            
            color = colors[dim % len(colors)]
            plt.scatter(v1, v2, alpha=0.3, s=15, edgecolors='none', color=color, label=f"Dim {dim} (r={corr:.2f})")
            
        if not has_data:
            print(f"No valid data found for plot {filename_suffix}")
            plt.close()
            return

        all_v1 = np.concatenate(reg_x)
        all_v2 = np.concatenate(reg_y)
        
        # Reference line
        # Use simple fixed range since user wants -2 to 2 anyway
        plt.plot([-2.5, 2.5], [-2.5, 2.5], 'k--', alpha=0.5, zorder=0, label='y=x')
        
        # Regression
        if len(all_v1) > 1:
            try:
                slope, intercept = np.polyfit(all_v1, all_v2, 1)
                print(f"Global Regression: y={slope:.4f}x + {intercept:.4f}")
                
                # Use fixed range for regression line display if we set manual limits, 
                # but let's stick to extending across the plot area
                # Since we are forcing x-axis to -2 to 2 later, we ensure line covers it
                x_vals = np.array([-2.2, 2.2]) 
                y_vals = slope * x_vals + intercept
                plt.plot(x_vals, y_vals, 'k-', linewidth=2, label=f'Reg: y={slope:.2f}x+{intercept:.2f}', zorder=10)
            except Exception as e:
                print(f"Could not fit regression line: {e}")

        plt.xlabel("Background Data SHAP")
        plt.ylabel("Safe State SHAP")
        plt.title(f"SHAP Correlation {title_suffix}")
        
        # Set fixed axis limits
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        f_pdf = os.path.join(OUTPUT_FOLDER, f"shap_correlation_{filename_suffix}.pdf")
        f_png = os.path.join(OUTPUT_FOLDER, f"shap_correlation_{filename_suffix}.png")
        plt.savefig(f_pdf, format='pdf')
        plt.savefig(f_png, format='png')
        plt.close()
        print(f"Saved {filename_suffix} to {OUTPUT_FOLDER}")

    # Generate standard plot
    create_correlation_plot(bg_array, safe_array, "combined", "")
    
    # Generate filtered plot
    create_correlation_plot(bg_array, safe_array, "combined_filtered", " (>0.3 Influence)", threshold=0.3)
    
    # --- Base Value Correlation Plot ---
    # Convert lists to arrays: (Total_Samples, n_outputs)
    # Each entry in base list is shape (n_outputs,)
    
    try:
        bg_base = np.vstack(all_bg_base)
        safe_base = np.vstack(all_safe_base)
    except Exception as e:
        print(f"Could not stack base values: {e}")
        return

    print("--- Generating Plot: Base Values ---")
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
    
    reg_x = []
    reg_y = []
    has_data = False
    
    # Plot each output dimension
    for dim in range(bg_base.shape[1]):
        v1 = bg_base[:, dim]
        v2 = safe_base[:, dim]
        
        valid_mask = ~np.isnan(v1) & ~np.isnan(v2)
        v1 = v1[valid_mask]
        v2 = v2[valid_mask]
        
        if len(v1) < 2:
            continue
            
        has_data = True
        reg_x.append(v1)
        reg_y.append(v2)
        
        corr, p_value = pearsonr(v1, v2)
        print(f"Base Value Dim {dim}: r={corr:.4f} (p={p_value:.4g})")
        
        color = colors[dim % len(colors)]
        plt.scatter(v1, v2, alpha=0.3, s=15, edgecolors='none', color=color, label=f"Dim {dim} (r={corr:.2f})")

    if has_data:
        all_v1 = np.concatenate(reg_x)
        all_v2 = np.concatenate(reg_y)
        
        # Reference line
        g_min = min(np.min(all_v1), np.min(all_v2))
        g_max = max(np.max(all_v1), np.max(all_v2))
        
        span = g_max - g_min
        if span == 0: span = 1.0
        padding = span * 0.05
        
        plt.plot([g_min-padding, g_max+padding], [g_min-padding, g_max+padding], 'k--', alpha=0.5, zorder=0, label='y=x')
        
        # Regression
        if len(all_v1) > 1:
            try:
                slope, intercept = np.polyfit(all_v1, all_v2, 1)
                print(f"Base Value Regression: y={slope:.4f}x + {intercept:.4f}")
                x_vals = np.array([g_min-padding, g_max+padding])
                y_vals = slope * x_vals + intercept
                plt.plot(x_vals, y_vals, 'k-', linewidth=2, label=f'Reg: y={slope:.2f}x+{intercept:.2f}', zorder=10)
            except Exception as e:
                print(f"Could not fit regression line: {e}")

    plt.xlabel("Background Data Base Value")
    plt.ylabel("Safe State Base Value")
    plt.title(f"SHAP Base Value Correlation")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    f_pdf = os.path.join(OUTPUT_FOLDER, "shap_correlation_base_values.pdf")
    f_png = os.path.join(OUTPUT_FOLDER, "shap_correlation_base_values.png")
    plt.savefig(f_pdf, format='pdf')
    plt.savefig(f_png, format='png')
    plt.close()
    print(f"Saved base value correlation to {OUTPUT_FOLDER}")

if __name__ == "__main__":
    main()
