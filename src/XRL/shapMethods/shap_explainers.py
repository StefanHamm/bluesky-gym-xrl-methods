
import numpy as np
import shap

def runSafeStateExplainer(model, observation,safe_vals,default_baseline=None):
    # 1. SETUP: We tell SHAP to explain features 0, 1, 2... (the intruders)
    number_of_aircrafts = len(observation[list(safe_vals.keys())[0]])
    # We pass indices [0, 1, 2...] as the "Input" to SHAP
    testX = np.array([np.arange(number_of_aircrafts)]) 

    # 2. MASKER: Returns simple modified INDICES
    # This keeps SHAP happy because it gets arrays it can concatenate.
    def cheat_masker(mask, X):
        # X is just [0, 1, 2...], mask is [True, False, True...]
        
        return [mask]

    # 3. MODEL WRAPPER: The "Real" Masker
    # This intercepts the array from SHAP, builds the dictionary, and calls your model.
    def custom_model_wrapper(X_batch):

        total_evals = len(X_batch)
        
        # Batch for efficiency
        # 1. Create a large batch of observations by replicating the original one
        obs_batch = {k: np.tile(v, (total_evals, 1)) for k, v in observation.items()}
        
        
        # 2. Vectorized or efficient update of the batch
        # Iterate over each mask (row in X_batch) and update corresponding obs in obs_batch
        for i, row_indices in enumerate(X_batch):
            # row_indices is the mask for the i-th observation in the batch
            # Find indices where intruder should be masked (value is 0/False)
            masked_indices = np.where(row_indices==0)[0]
            
            if len(masked_indices) > 0:
                for key in safe_vals.keys():
                    obs_batch[key][i, masked_indices] = safe_vals[key]

        # 3. Batch Predict
        # Single call for all permutations
        pred, _ = model.predict(obs_batch, deterministic=True)
        if default_baseline is not None:
            # Adjust the baseline prediction to be the passed default baseline
            #index where all groups are masked can be done where the sum is 0
            for i, row_indices in enumerate(X_batch):
                if np.sum(row_indices) == 0:
                    pred[i] = default_baseline
       
        return np.array(pred)

    # 4. RUN
    # Note: We pass the WRAPPER as the model, and cheat_masker as the masker
    explainer = shap.explainers.Exact(custom_model_wrapper, cheat_masker)
    
    shap_values = explainer(testX)
    #shap.plots.bar(shap_values)
    return shap_values


def runBackgroundExplainer(model,observation,backgroundData,mapping:list,n_samples=50):
    # runs shap permutation explainer on the given observation
    # returns shap values for the observation
    # this is done using the background data to sample from to displace  single intruders in the observation
    
    number_of_intruders = len(observation[mapping[0]])
    # We pass indices [0, 1, 2...] as the "Input" to SHAP
    testX = np.array([np.arange(number_of_intruders)]) 

    # Masker: Returns multiple copies of the mask to allow averaging over background samples
    def cheat_masker(mask,X):
        return [mask]

    # Model Wrapper
    def custom_model_wrapper(X_batch):
        #print(len(X_batch))
        # Optimized: Batched creation and prediction
        n_masks = len(X_batch)
        total_evals = n_masks * n_samples
        
        # 1. Create a batch of observations by repeating the original observation
        # obs_batch = {key: (total_evals, features...)}
        obs_batch = {k: np.tile(v, (total_evals, 1)) for k, v in observation.items()}

        # 2. Vectorized filling of background data
        for m, mask_row in enumerate(X_batch):
            # Find indices of intruders that are masked (False)
            masked_indices = np.where(~np.array(mask_row, dtype=bool))[0]
            
            if len(masked_indices) > 0:
                # Determine the slice of rows corresponding to this mask
                start_idx = m * n_samples
                end_idx = start_idx + n_samples
                
                for intruder_idx in masked_indices:
                    # Generate random indices for background samples
                    rand_idxs = np.random.randint(0, len(backgroundData), size=n_samples)
                    samples = backgroundData[rand_idxs]
                    
                    for i, feature in enumerate(mapping):
                        obs_batch[feature][start_idx:end_idx, intruder_idx] = samples[:, i]
        
        # 3. Batch prediction (Single call to model, much faster)
        preds, _ = model.predict(obs_batch, deterministic=True)
        
        # 4. Reshape and Average
        # preds shape: (total_evals, output_dim) -> reshape to (n_masks, n_samples, output_dim)
        # Then average over the n_samples dimension to get expected value per mask
        preds_reshaped = preds.reshape(n_masks, n_samples, -1)
        return preds_reshaped.mean(axis=1)

    explainer = shap.explainers.Exact(custom_model_wrapper, cheat_masker)
    
    shap_values = explainer(testX)
    return shap_values

DESCENT_MAPPING = ["altitude", "vz", "target_altitude", "runway_distance"]

def get_background_data_descent(env, model, seed=42, n_samples=1000):
    """Collects empirical background data with deterministic initialization."""
    obs_list = []
    
    # Explicitly seed the first trajectory
    obs, _ = env.reset(seed=seed)
    
    for _ in range(n_samples):
        flat_features = [float(obs[key][0]) for key in DESCENT_MAPPING]
        obs_list.append(flat_features)
        
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            # Allow PRNG to continue normally to explore different trajectories
            obs, _ = env.reset()
            
    return np.array(obs_list)

def runFeatureBackgroundExplainer(model, obs_dict, background_summary):
    """Executes KernelSHAP over the 4 kinematic variables of DescentEnv-v0."""
    
    # KernelExplainer requires a function that maps a 2D matrix to predictions
    def predict_wrapper(X_matrix):
        # Reconstruct the SB3 Dictionary Observation Space from the matrix
        batch_obs = {
            "altitude": X_matrix[:, 0].reshape(-1, 1),
            "vz": X_matrix[:, 1].reshape(-1, 1),
            "target_altitude": X_matrix[:, 2].reshape(-1, 1),
            "runway_distance": X_matrix[:, 3].reshape(-1, 1)
        }
        actions, _ = model.predict(batch_obs, deterministic=True)
        return actions

    explainer = shap.KernelExplainer(predict_wrapper, background_summary)
    
    # Flatten current observation into a 1x4 matrix
    obs_array = np.array([[float(obs_dict[key][0]) for key in DESCENT_MAPPING]])
    
    shap_vals_raw = explainer.shap_values(obs_array, silent=True)
    
    base_val = explainer.expected_value
    if isinstance(base_val, np.ndarray):
        base_val = base_val[0]
        
    shap_vals_arr = shap_vals_raw[0] if isinstance(shap_vals_raw, list) else shap_vals_raw

    return shap.Explanation(
        values=shap_vals_arr,
        base_values=np.array([[base_val]]),
        data=obs_array
    )

STATIC_OBS_KEYS = [
    "restricted_area_radius", 
    "restricted_area_distance", 
    "cos_difference_restricted_area_pos", 
    "sin_difference_restricted_area_pos"
]

def get_background_data_static(env, model, seed=42, n_samples=500):
    """Collects empirical background data for the 10 static obstacles."""
    obs_list = []
    obs, _ = env.reset(seed=seed)
    for _ in range(n_samples):
        # Extract features for all 10 obstacles
        # Each key yields a 1D array of shape (10,)
        flat_features = [obs[key] for key in STATIC_OBS_KEYS]
        # Stack into (10, 4) matrix representing the 10 obstacles
        obs_list.append(np.column_stack(flat_features))
        
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
            
    # Flattens into a 2D matrix of shape (n_samples * 10, 4)
    return np.vstack(obs_list)

def runStaticObstacleExplainer(model, observation, backgroundData, n_samples=50):
    """Multi-output Exact SHAP explainer for Heading and Speed."""
    number_of_obstacles = 10 
    testX = np.array([np.arange(number_of_obstacles)]) 

    def cheat_masker(mask, X):
        return [mask]

    def custom_model_wrapper(X_batch):
        n_masks = len(X_batch)
        total_evals = n_masks * n_samples
        
        # Tile the base observation to batch inference
        obs_batch = {k: np.tile(v, (total_evals, 1)) for k, v in observation.items()}

        for m, mask_row in enumerate(X_batch):
            masked_indices = np.where(~np.array(mask_row, dtype=bool))[0]
            if len(masked_indices) > 0:
                start_idx = m * n_samples
                end_idx = start_idx + n_samples
                
                for obs_idx in masked_indices:
                    # Sample random background obstacles
                    rand_idxs = np.random.randint(0, len(backgroundData), size=n_samples)
                    samples = backgroundData[rand_idxs] # Shape (n_samples, 4)
                    
                    for i, feature in enumerate(STATIC_OBS_KEYS):
                        obs_batch[feature][start_idx:end_idx, obs_idx] = samples[:, i]
        
        preds, _ = model.predict(obs_batch, deterministic=True)
        # Reshape to (n_masks, n_samples, 2 outputs) and average over samples
        preds_reshaped = preds.reshape(n_masks, n_samples, 2)
        return preds_reshaped.mean(axis=1)

    explainer = shap.explainers.Exact(custom_model_wrapper, cheat_masker)
    shap_values = explainer(testX)
    
    return shap_values

def get_background_data_merge(env, model, seed=42, n_samples=50):
    """Collects empirical background dictionary observations for MergeEnv."""
    obs_list = []
    obs, _ = env.reset(seed=seed)
    for _ in range(n_samples):
        # Deep copy the observation dictionary to avoid reference mutation
        obs_list.append({k: np.copy(v) for k, v in obs.items()})
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
    return obs_list

def runMergeExplainer(model, observation, backgroundData, n_samples=50):
    """Multi-output Exact SHAP explainer handling 5 intruders and 3 global features."""
    M = 8  # 5 Intruders + 1 Routing(Drift) + 1 Longitudinal(Dist) + 1 Kinematic(Spd)
    testX = np.array([np.arange(M)])

    def cheat_masker(mask, X):
        return [mask]

    def custom_model_wrapper(X_batch):
        n_masks = len(X_batch)
        total_evals = n_masks * n_samples
        
        # Tile base observation for batch inference
        obs_batch = {k: np.tile(v, (total_evals, 1)) for k, v in observation.items()}

        for m, mask_row in enumerate(X_batch):
            start_idx = m * n_samples
            
            # Sample random background indices
            rand_idxs = np.random.randint(0, len(backgroundData), size=n_samples)

            # Coalition 0-4 (Intruders)
            for i in range(5):
                if not mask_row[i]:
                    for key in ["x_r", "y_r", "vx_r", "vy_r", "cos(track)", "sin(track)", "distances"]:
                        for s, bg_idx in enumerate(rand_idxs):
                            obs_batch[key][start_idx + s, i] = backgroundData[bg_idx][key].flatten()[i]

            # Coalition 5 (Routing / Drift)
            if not mask_row[5]:
                for key in ["cos(drift)", "sin(drift)"]:
                    for s, bg_idx in enumerate(rand_idxs):
                        obs_batch[key][start_idx + s, 0] = backgroundData[bg_idx][key].flatten()[0]

            # Coalition 6 (Longitudinal / Distance)
            if not mask_row[6]:
                for key in ["waypoint_dist", "faf_reached"]:
                    for s, bg_idx in enumerate(rand_idxs):
                        obs_batch[key][start_idx + s, 0] = backgroundData[bg_idx][key].flatten()[0]

            # Coalition 7 (Kinematic / Airspeed)
            if not mask_row[7]:
                for s, bg_idx in enumerate(rand_idxs):
                    obs_batch["airspeed"][start_idx + s, 0] = backgroundData[bg_idx]["airspeed"].flatten()[0]

        preds, _ = model.predict(obs_batch, deterministic=True)
        # Reshape to (n_masks, n_samples, 2 outputs) and average over background samples
        preds_reshaped = preds.reshape(n_masks, n_samples, 2)
        return preds_reshaped.mean(axis=1)

    explainer = shap.explainers.Exact(custom_model_wrapper, cheat_masker)
    return explainer(testX)
