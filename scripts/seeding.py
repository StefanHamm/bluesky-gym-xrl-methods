import gymnasium as gym
import numpy as np
import bluesky_gym
import time

# Register environments
bluesky_gym.register_envs()

ENV_IDS = [
    "StaticObstacleEnv-v0",
    "DescentEnv-v0",
    "PlanWaypointEnv-v0",
    "HorizontalCREnv-v0",
    "VerticalCREnv-v0",
    "SectorCREnv-v0",
    "MergeEnv-v0",
]

def collect_initial_observations(env_id, seed=None, episodes=3):
    """
    Creates an environment, resets it multiple times (with the given seed if provided),
    and collects the initial observations.
    """
    observations = []
    # Create a fresh environment instance
    env = gym.make(env_id,render_mode="human")
    
    try:
        for i in range(episodes):
            # For the first reset, we use the provided seed. 
            # For subsequent resets, we don't pass the seed to let the RNG continue its sequence.
            if i == 0:
                obs, _ = env.reset(seed=seed)
            else:
                obs, _ = env.reset()
            time.sleep(1)  # Allow time for rendering to complete    
            observations.append(obs)
            
    finally:
        
        env.close()
        
    return observations

def compare_observations(obs_list1, obs_list2):
    """Returns True if observations are identical."""
    if len(obs_list1) != len(obs_list2):
        return False
    
    for o1, o2 in zip(obs_list1, obs_list2):
        if not are_observations_equal(o1, o2):
            return False
                
    return True

def are_observations_equal(o1, o2):
    """Helper to compare two single observations."""
    if isinstance(o1, dict) and isinstance(o2, dict):
        if o1.keys() != o2.keys():
            return False
        for key in o1:
            val1 = o1[key]
            val2 = o2[key]
            if isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
                if not np.allclose(val1, val2, atol=1e-8):
                    return False
            else:
                if val1 != val2:
                    return False
        return True
    elif isinstance(o1, np.ndarray) and isinstance(o2, np.ndarray):
        return np.allclose(o1, o2, atol=1e-8)
    else:
        return not np.any(o1 != o2)

def check_internal_diversity(obs_list):
    """Returns True if the observations in the list are not all identical."""
    if len(obs_list) < 2:
        return True
    
    first_obs = obs_list[0]
    for i in range(1, len(obs_list)):
        if not are_observations_equal(first_obs, obs_list[i]):
            return True
            
    return False

def test_environment_seeding(env_id):
    print(f"Testing seeding for {env_id}...")
    
    seed_val = 42
    other_seed = 123
    episodes = 3
    
    # 1. Env 1 with seed
    print("  Creating Env 1 (Seed 42)")
    obs1 = collect_initial_observations(env_id, seed=seed_val, episodes=episodes)
    
    # 2. Env 2 with same seed
    print("  Creating Env 2 (Seed 42)")
    obs2 = collect_initial_observations(env_id, seed=seed_val, episodes=episodes)
    
    # 3. Env 3 with different seed
    print("  Creating Env 3 (Seed 123)")
    obs3 = collect_initial_observations(env_id, seed=other_seed, episodes=episodes)
    
    # 4. Env 4 with no seed
    print("  Creating Env 4 (No Seed)")
    obs4 = collect_initial_observations(env_id, seed=None, episodes=episodes)
    
    # 5. Env 5 with no seed
    print("  Creating Env 5 (No Seed)")
    obs5 = collect_initial_observations(env_id, seed=None, episodes=episodes)
    
    # Checks
    
    passed = True
    
    # Check 1: obs1 == obs2 (Same seed -> Identical)
    # Verify that subsequent episodes are identical too (Deterministic sequence)
    print("  Verifying deterministic sequence (same seed):")
    deterministic = True
    if len(obs1) != len(obs2):
        print(f"    [ERROR] Episode counts differ: {len(obs1)} vs {len(obs2)}")
        deterministic = False
    else:
        for i in range(len(obs1)):
            match = are_observations_equal(obs1[i], obs2[i])
            status = "MATCH" if match else "MISMATCH"
            print(f"    Episode {i+1}: {status}") 
            if not match:
                deterministic = False

    if deterministic:
        print(f"  [PASS] Identical seeds produced identical sequences of episodes.")
    else:
        print(f"  [FAIL] Identical seeds produced different sequences of episodes.")
        passed = False

    # Check 2: obs1 != obs3 (Different seed -> Different)
    if not compare_observations(obs1, obs3):
        print(f"  [PASS] Different seeds produced different sequences.")
    else:
        print(f"  [WARN] Different seeds produced identical sequences. (Might be chance or static env)")
        
    # Check 3: obs4 != obs5 (No seed -> Different/Random)
    if not compare_observations(obs4, obs5):
        print(f"  [PASS] Two unseeded environments produced different sequences.")
    else:
        print(f"  [FAIL] Two unseeded environments produced identical sequences. (Expected random behavior)")
        passed = False

    # Check 4: Internal Randomness check (If no seed is set, each episode should be random)
    # We check if episodes within one unseeded run are identical.
    if check_internal_diversity(obs4):
        print(f"  [PASS] Unseeded environment produced diverse episodes (internal randomness).")
    else:
        print(f"  [WARN] Unseeded environment produced identical episodes. (Could be a static environment)")
        
    if passed:
        print(f"  {env_id} seeding test completed successfully.\n")
    else:
        print(f"  {env_id} seeding test FAILED.\n")
        
    return passed

if __name__ == "__main__":
    all_passed = True
    print(f"Starting seeding tests for {len(ENV_IDS)} environments: {ENV_IDS}\n")
    
    for env_id in ENV_IDS:
        try:
            passed = test_environment_seeding(env_id)
            if not passed:
                all_passed = False
        except Exception as e:
            print(f"  [ERROR] Exception while testing {env_id}: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
            
    if all_passed:
        print("All environments passed seeding tests.")
    else:
        print("Some seeding tests failed.")