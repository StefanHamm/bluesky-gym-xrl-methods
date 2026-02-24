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
		
	return np.median(returns), returns

if __name__ == "__main__":
	JOBID = "4901832"
	NUM_EPISODES = 300
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
 
	combined_modelpath = rf"models\{JOBID}\PlanWaypointEvadeEnv-v0\PlanWaypointEvadeEnv-v0_SAC_vecEnvLogs_baseline_model_mp.zip"
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

		median_return_native, returns_native = future_native.result()
		median_return_threshold, returns_threshold = future_threshold.result()
		median_return_blending, returns_blending = future_blending.result()
		median_return_fatigue, returns_fatigue = future_fatigue.result()
		median_return_predictive, returns_predictive = future_predictive.result()

	print("\n" * 5) # Clear lines after tqdm progress bars
	print(f"Native SAC median return: {median_return_native}")
	print(f"ThresholdGating median return: {median_return_threshold}")
	print(f"BlendingGating median return: {median_return_blending}")
	print(f"FatigueBlendingGating median return: {median_return_fatigue}")
	print(f"PredictiveShieldingGating median return: {median_return_predictive}")

	

	# Plot results
	strategies = [
		"ThresholdGating",
		"BlendingGating",
		"FatigueBlendingGating",
		"PredictiveShieldingGating",
		"Native SAC"
	]
	medians = [
		median_return_threshold,
		median_return_blending,
		median_return_fatigue,
		median_return_predictive,
		median_return_native
	]
	
	all_returns = [
		returns_threshold,
		returns_blending,
		returns_fatigue,
		returns_predictive,
		returns_native
	]
	
	means = [np.mean(r) for r in all_returns]
	q25 = [np.percentile(r, 25) for r in all_returns]
	q75 = [np.percentile(r, 75) for r in all_returns]
	
	yerr_lower = [m - q for m, q in zip(means, q25)]
	yerr_upper = [q - m for m, q in zip(means, q75)]

	colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
	
	# Plot 1: Median Returns (Point plot to better compare negative values)
	for i, (strategy, median) in enumerate(zip(strategies, medians)):
		ax1.plot(strategy, median, marker='o', markersize=10, color=colors[i])
	ax1.set_ylabel("Median Return")
	ax1.set_title(f"Median Returns of Strategies ({NUM_EPISODES} episodes)")
	ax1.tick_params(axis='x', rotation=20)
	ax1.grid(True, linestyle='--', alpha=0.7)

	# Plot 2: Mean Returns with IQR error bars
	for i, (strategy, mean, yl, yu) in enumerate(zip(strategies, means, yerr_lower, yerr_upper)):
		ax2.errorbar(strategy, mean, yerr=[[yl], [yu]], fmt='o', capsize=5, markersize=10, color=colors[i], ecolor='black')
	ax2.set_ylabel("Mean Return")
	ax2.set_title(f"Mean Returns with IQR (25th-75th percentile)")
	ax2.tick_params(axis='x', rotation=20)
	ax2.grid(True, linestyle='--', alpha=0.7)

	plt.tight_layout()

	# Save the plot under plots/JOBID
	import os
	save_dir = os.path.join("plots", JOBID)
	os.makedirs(save_dir, exist_ok=True)
	
	# Export data using pandas
	df = pd.DataFrame({
		"ThresholdGating": returns_threshold,
		"BlendingGating": returns_blending,
		"FatigueBlendingGating": returns_fatigue,
		"PredictiveShieldingGating": returns_predictive,
		"Native SAC": returns_native
	})
	csv_path = os.path.join(save_dir, "moe_evaluation_results.csv")
	df.to_csv(csv_path, index=False)
	print(f"Data exported to {csv_path}")

	save_path = os.path.join(save_dir, "moe_returns_comparison.png")
	plt.savefig(save_path)

	plt.show()

