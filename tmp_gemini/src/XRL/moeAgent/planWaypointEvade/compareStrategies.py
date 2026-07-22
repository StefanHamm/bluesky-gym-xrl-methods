from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

import gymnasium as gym
from stable_baselines3 import SAC
import bluesky_gym
import numpy as np
from src.XRL.moeAgent.controlEvade import ThresholdGating, BlendingGating, FatigueBlendingGating, PredictiveShieldingGating
from bluesky_gym.envs.free_flight_env import SENSOR_RANGE
from bluesky_gym.utils.constants import NM2KM

bluesky_gym.register_envs()

def metric_extractor(obs):
	return np.min(obs["intruder_distance"])

def evaluate_strategy(env_name, model, episodes=100, use_gating=False, gating_model=None, seed=42, desc="Episodes", position=0):
	"""
	Evaluate a model or gating strategy in the given environment.

	Args:
		env_name (str): Name of the environment.
		model: For the native SAC evaluation (use_gating=False), this should be the model trained on the combined environment (PlanWaypointEvadeEnv-v0). For gating strategies, this is the control model.
		episodes (int): Number of episodes to run.
		use_gating (bool): Whether to use a gating strategy.
		gating_model: The gating model to use if use_gating is True.
		seed (int): Random seed.
	"""
	returns = []
	waypoints = []
	intrusions = []
	import gymnasium as gym
	env = gym.make(env_name, training= not use_gating) #removes the observation of heading for native model
	for ep in tqdm(range(episodes), desc=desc, position=position, leave=True):
		obs, info = env.reset(seed=seed+ep)

		done = truncated = False
		ep_return = 0
		timestep = 0
		while not (done or truncated):
			timestep += 1
			if use_gating:
				action, evading = gating_model.predict(obs, deterministic=True)
				env.unwrapped.update_evading_status(evading)
			else:
				action, _ = model.predict(obs, deterministic=True)
			obs, reward, done, truncated, info = env.step(action)
			
			ep_return += reward
			if timestep >= 900:
				truncated = True
		
		returns.append(ep_return)
		waypoints.append(info.get('waypoints_completed', 0))
		intrusions.append(info.get('total_intrusions', 0))
		
	return np.median(returns), returns, waypoints, intrusions


def styled_boxplot(ax, data, labels, colors, title, ylabel):
	bp = ax.boxplot(
		data,
		patch_artist=True,
		showmeans=True,
		meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='white', markersize=7),
		medianprops=dict(color='black', linewidth=2),
		flierprops=dict(marker='o', markersize=3, alpha=0.4),
		whiskerprops=dict(linewidth=1.5),
		capprops=dict(linewidth=1.5),
	)
	for patch, color in zip(bp['boxes'], colors):
		patch.set_facecolor(color)
		patch.set_alpha(0.7)
	for flier, color in zip(bp['fliers'], colors):
		flier.set_markerfacecolor(color)
		flier.set_markeredgecolor(color)
	ax.set_xticks(range(1, len(labels) + 1))
	ax.set_xticklabels(labels, rotation=20, ha='right')
	ax.set_title(title)
	ax.set_ylabel(ylabel)
	ax.grid(True, linestyle='--', alpha=0.7)

	from matplotlib.lines import Line2D
	legend_elements = [
		Line2D([0], [0], color='black', linewidth=2, label='Median'),
		Line2D([0], [0], marker='D', color='w', markeredgecolor='black', markerfacecolor='white', markersize=7, label='Mean'),
	]
	ax.legend(handles=legend_elements, frameon=True)


if __name__ == "__main__":
	JOBID = "4901832"
	NUM_EPISODES = 150
	env_name = 'PlanWaypointEvadeEnv-v0'
	
	control_modelpath = rf"models\{JOBID}\PlanWaypointEnv-v2\PlanWaypointEnv-v2_SAC_vecEnvLogs_baseline_model_mp.zip"
	control_model = SAC.load(control_modelpath, device='cpu')
	control_keywords = [
		"waypoint_distance",
		"cos_difference",
		"sin_difference",
		"waypoint_reached",
		"previous_action"
	]
	evade_modelpath = rf"models\{JOBID}\FreeFlightCREnv-v0\FreeFlightCREnv-v0_SAC_vecEnvLogs_baseline_model_mp.zip"
	evade_model = SAC.load(evade_modelpath, device='cpu')
	evade_keywords = [
		"intruder_distance",
		"cos_difference_pos",
		"sin_difference_pos",
		"x_difference_speed",
		"y_difference_speed",
		"cos_own_heading",
		"sin_own_heading"
	]
 
	COMBINED_JOBID = "5124610"
 
	combined_modelpath = rf"models\{COMBINED_JOBID}\PlanWaypointEvadeEnv-v0\PlanWaypointEvadeEnv-v0_SAC_vecEnvLogs_baseline_model_mp.zip"
	combined_model = SAC.load(combined_modelpath, device='cpu')
 
	min_nm = 7
	max_nm = 15
	min_normed = min_nm * NM2KM / SENSOR_RANGE
	max_normed = max_nm * NM2KM / SENSOR_RANGE

	# ThresholdGating
	threshold_gating = ThresholdGating(
		controlModel=control_model,
		evadeModel=evade_model,
		controlKeys=control_keywords,
		evadeKeys=evade_keywords,
		threshold=min_normed,
		metric_extractor=metric_extractor
	)

	# BlendingGating
	blending_gating = BlendingGating(
		controlModel=control_model,
		evadeModel=evade_model,
		controlKeys=control_keywords,
		evadeKeys=evade_keywords,
		min_val=min_normed,
		max_val=max_normed,
		metric_extractor=metric_extractor
	)

	# FatigueBlendingGating
	fatigue_gating = FatigueBlendingGating(
		controlModel=control_model,
		evadeModel=evade_model,
		controlKeys=control_keywords,
		evadeKeys=evade_keywords,
		min_val=min_normed,
		max_val=max_normed,
		fatigue_rate=0.005,
		max_fatigue=0.3,
		metric_extractor=metric_extractor
	)

	# PredictiveShieldingGating
	predictive_gating = PredictiveShieldingGating(
		controlModel=control_model,
		evadeModel=evade_model,
		controlKeys=control_keywords,
		evadeKeys=evade_keywords,
		number_of_future_steps=240,
		number_intrusor_aircraft=5,
		intrusion_distance_nm=5,
		alpha_update_interval=5
	)

	import concurrent.futures

	print("Starting evaluations in parallel...")
	with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
		future_native = executor.submit(evaluate_strategy, env_name, combined_model, NUM_EPISODES, False, None, 42, "Native SAC", 0)
		future_threshold = executor.submit(evaluate_strategy, env_name, control_model, NUM_EPISODES, True, threshold_gating, 42, "ThresholdGating", 1)
		future_blending = executor.submit(evaluate_strategy, env_name, control_model, NUM_EPISODES, True, blending_gating, 42, "BlendingGating", 2)
		future_fatigue = executor.submit(evaluate_strategy, env_name, control_model, NUM_EPISODES, True, fatigue_gating, 42, "FatigueGating", 3)
		future_predictive = executor.submit(evaluate_strategy, env_name, control_model, NUM_EPISODES, True, predictive_gating, 42, "PredictiveGating", 4)

		median_return_native, returns_native, waypoints_native, intrusions_native = future_native.result()
		median_return_threshold, returns_threshold, waypoints_threshold, intrusions_threshold = future_threshold.result()
		median_return_blending, returns_blending, waypoints_blending, intrusions_blending = future_blending.result()
		median_return_fatigue, returns_fatigue, waypoints_fatigue, intrusions_fatigue = future_fatigue.result()
		median_return_predictive, returns_predictive, waypoints_predictive, intrusions_predictive = future_predictive.result()

	print("\n" * 5) # Clear lines after tqdm progress bars
	print(f"Native SAC median return: {median_return_native}")
	print(f"ThresholdGating median return: {median_return_threshold}")
	print(f"BlendingGating median return: {median_return_blending}")
	print(f"FatigueBlendingGating median return: {median_return_fatigue}")
	print(f"PredictiveShieldingGating median return: {median_return_predictive}")

	strategies = [
		"ThresholdGating",
		"BlendingGating",
		"FatigueBlendingGating",
		"PredictiveShieldingGating",
		"Native SAC"
	]

	all_returns = [
		returns_threshold,
		returns_blending,
		returns_fatigue,
		returns_predictive,
		returns_native
	]
	
	all_waypoints = [
		waypoints_threshold,
		waypoints_blending,
		waypoints_fatigue,
		waypoints_predictive,
		waypoints_native
	]
	
	all_intrusions = [
		intrusions_threshold,
		intrusions_blending,
		intrusions_fatigue,
		intrusions_predictive,
		intrusions_native
	]

	colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

	import os
	save_dir = os.path.join("plots", f"{COMBINED_JOBID}")
	os.makedirs(save_dir, exist_ok=True)

	# Figure 1: Returns
	fig1, ax1 = plt.subplots(figsize=(8, 6))
	styled_boxplot(ax1, all_returns, strategies, colors,
				   f"Episode Returns ({NUM_EPISODES} episodes)", "Episode Return")
	plt.tight_layout()
	plt.savefig(os.path.join(save_dir, "boxplot_returns.png"), bbox_inches='tight')

	# Figure 2: Waypoints Completed
	fig2, ax2 = plt.subplots(figsize=(8, 6))
	styled_boxplot(ax2, all_waypoints, strategies, colors,
				   f"Waypoints Completed ({NUM_EPISODES} episodes)", "Waypoints Completed")
	plt.tight_layout()
	plt.savefig(os.path.join(save_dir, "boxplot_waypoints.png"), bbox_inches='tight')

	# Figure 3: Total Intrusions
	fig3, ax3 = plt.subplots(figsize=(8, 6))
	styled_boxplot(ax3, all_intrusions, strategies, colors,
				   f"Total Intrusions ({NUM_EPISODES} episodes)", "Intrusions")
	plt.tight_layout()
	plt.savefig(os.path.join(save_dir, "boxplot_intrusions.png"), bbox_inches='tight')

	print(f"Plots saved to {save_dir}")

	# Export data using pandas
	df = pd.DataFrame({
		"ThresholdGating_Return": returns_threshold,
		"BlendingGating_Return": returns_blending,
		"FatigueBlendingGating_Return": returns_fatigue,
		"PredictiveShieldingGating_Return": returns_predictive,
		"NativeSAC_Return": returns_native,
		"ThresholdGating_Waypoints": waypoints_threshold,
		"BlendingGating_Waypoints": waypoints_blending,
		"FatigueBlendingGating_Waypoints": waypoints_fatigue,
		"PredictiveShieldingGating_Waypoints": waypoints_predictive,
		"NativeSAC_Waypoints": waypoints_native,
		"ThresholdGating_Intrusions": intrusions_threshold,
		"BlendingGating_Intrusions": intrusions_blending,
		"FatigueBlendingGating_Intrusions": intrusions_fatigue,
		"PredictiveShieldingGating_Intrusions": intrusions_predictive,
		"NativeSAC_Intrusions": intrusions_native,
	})
	csv_path = os.path.join(save_dir, "moe_evaluation_results.csv")
	df.to_csv(csv_path, index=False)
	print(f"Data exported to {csv_path}")

	plt.show()