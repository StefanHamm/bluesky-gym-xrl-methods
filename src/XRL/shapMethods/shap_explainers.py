
import numpy as np
import shap

def runSafeStateExplainer(model, observation,safe_vals):
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