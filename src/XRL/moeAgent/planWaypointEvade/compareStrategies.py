from tqdm import tqdm
import matplotlib.pyplot as plt

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

def evaluate_strategy(env, model, episodes=100, use_gating=False, gating_model=None, seed=42):
	returns = []
	for ep in tqdm(range(episodes), desc="Episodes"):
		done = truncated = False
		obs, info = env.reset(seed=seed)
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
	env_name = 'PlanWaypointEvadeEnv-v0'
	env = gym.make(env_name, training=False)
	env.reset(seed=42)
	control_modelpath = r"models\4901832\PlanWaypointEnv-v2\PlanWaypointEnv-v2_SAC_vecEnvLogs_baseline_model_mp.zip"
	control_model = SAC.load(control_modelpath, device='cpu')
	control_keywords = [
		"waypoint_distance",
		"cos_difference",
		"sin_difference",
		"waypoint_reached",
		"previous_action"
	]
	evade_modelpath = r"models\4901832\FreeFlightCREnv-v0\FreeFlightCREnv-v0_SAC_vecEnvLogs_baseline_model_mp.zip"
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

	print("Evaluating ThresholdGating...")
	median_return_threshold, returns_threshold = evaluate_strategy(env, control_model, episodes=100, use_gating=True, gating_model=threshold_gating, seed=42)
	print(f"ThresholdGating median return: {median_return_threshold}")

	env.reset(seed=42)
	print("Evaluating BlendingGating...")
	median_return_blending, returns_blending = evaluate_strategy(env, control_model, episodes=100, use_gating=True, gating_model=blending_gating, seed=42)
	print(f"BlendingGating median return: {median_return_blending}")

	env.reset(seed=42)
	print("Evaluating FatigueBlendingGating...")
	median_return_fatigue, returns_fatigue = evaluate_strategy(env, control_model, episodes=100, use_gating=True, gating_model=fatigue_gating, seed=42)
	print(f"FatigueBlendingGating median return: {median_return_fatigue}")

	env.reset(seed=42)
	print("Evaluating PredictiveShieldingGating...")
	median_return_predictive, returns_predictive = evaluate_strategy(env, control_model, episodes=100, use_gating=True, gating_model=predictive_gating, seed=42)
	print(f"PredictiveShieldingGating median return: {median_return_predictive}")

	env.reset(seed=42)
	print("Evaluating native SAC model...")
	median_return_native, returns_native = evaluate_strategy(env, control_model, episodes=100, use_gating=False, seed=42)
	print(f"Native SAC median return: {median_return_native}")

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
	plt.figure(figsize=(8, 5))
	plt.bar(strategies, medians, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"])
	plt.ylabel("Median Return")
	plt.title("Median Returns of Strategies (100 episodes)")
	plt.xticks(rotation=20)
	plt.tight_layout()
	plt.show()

