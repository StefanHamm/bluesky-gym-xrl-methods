import numpy as np
from abc import ABC, abstractmethod
from stable_baselines3 import SAC,PPO,TD3,DDPG,A2C

import numpy as np
from abc import ABC, abstractmethod
from stable_baselines3 import SAC,PPO,TD3,DDPG,A2C

import bluesky as bs

class BaseGatingStrategy(ABC):
    def __init__(self, controlModel: object, evadeModel: object, controlKeys: list[str], evadeKeys: list[str]):
        """
        Base class for Gating Strategies that manage two models (Control and Evade).
        
        :param controlModel: The model used for control (must be a Stable Baselines3 model).
        :param evadeModel: The model used for evasion (must be a Stable Baselines3 model).
        :param controlKeys: List of observation keys required by the control model.
        :param evadeKeys: List of observation keys required by the evade model.
        """
        self.controlModel = controlModel
        self.evadeModel = evadeModel
        self.controlKeys = controlKeys
        self.evadeKeys = evadeKeys
        
        # Validating model types
        allowed_types = (SAC, PPO, TD3, DDPG, A2C)
        assert isinstance(controlModel, allowed_types), f"Control model must be one of {allowed_types}"
        assert isinstance(evadeModel, allowed_types), f"Evade model must be one of {allowed_types}"

    def _get_sub_actions(self, obs, deterministic=True):
        """
        Helper to get actions from both models.
        """
        # Extract specific observations for each model
        control_obs = {key: obs[key] for key in self.controlKeys}
        evade_obs = {key: obs[key] for key in self.evadeKeys}
       
        
        # Get actions
        control_action, _ = self.controlModel.predict(control_obs, deterministic=deterministic)
        evade_action, _ = self.evadeModel.predict(evade_obs, deterministic=deterministic)
        
        return control_action, evade_action

    @abstractmethod
    def predict(self, obs, deterministic=True):
        """
        Determines the final action based on observations.
        To be implemented by subclasses.
        """
        pass

class ThresholdGating(BaseGatingStrategy):
    """
    Simple on/off switch. Uses evade action if metric <= threshold, else control action.
    """
    def __init__(self, controlModel, evadeModel, controlKeys, evadeKeys, threshold, metric_extractor):
        """
        :param threshold: The value at which to switch strategies.
        :param metric_extractor: A function that takes `obs` and returns a numeric metric (e.g. distance).
        """
        super().__init__(controlModel, evadeModel, controlKeys, evadeKeys)
        self.threshold = threshold
        self.metric_extractor = metric_extractor

    def predict(self, obs, deterministic=True):
        # Get actions from both models
        control_action, evade_action = self._get_sub_actions(obs, deterministic)
        
        metric = self.metric_extractor(obs)
        
        # Boolean mask: True where we should use evade action
        mask = metric <= self.threshold
        # Handle scalar case
        if np.isscalar(mask):
            final_action = evade_action if mask else control_action
        else:
            # Handle vectorized case
            if len(control_action.shape) > 1:
                mask_reshaped = mask.reshape(-1, 1)
            else:
                mask_reshaped = mask
            final_action = np.where(mask_reshaped, evade_action, control_action)
            
        return final_action, mask

class BlendingGating(BaseGatingStrategy):
    """
    Linearly blends between control and evade actions based on a metric.
    """
    def __init__(self, controlModel, evadeModel, controlKeys, evadeKeys, min_val, max_val, metric_extractor, inverted=True):
        """
        :param min_val: Metric value at one end of the spectrum.
        :param max_val: Metric value at the other end.
        :param metric_extractor: Function to extract metric from obs.
        :param inverted: 
            If True (default for distance): 
                metric <= min_val -> 100% Evade
                metric >= max_val -> 100% Control
        """
        super().__init__(controlModel, evadeModel, controlKeys, evadeKeys)
        self.min_val = min_val
        self.max_val = max_val
        self.metric_extractor = metric_extractor
        self.inverted = inverted

    def predict(self, obs, deterministic=True):
        # Get actions from both models
        control_action, evade_action = self._get_sub_actions(obs, deterministic)
        #print(f"Control Action: {control_action}, Evade Action: {evade_action}")
        
        metric = self.metric_extractor(obs)
        
        # Clip metric to range and normalize to [0, 1]
        clipped_metric = np.clip(metric, self.min_val, self.max_val)
        norm_position = (clipped_metric - self.min_val) / (self.max_val - self.min_val)
        
        # Calculate alpha (weight for evade action)
        if self.inverted:
            # Low metric (distance) = High evade weight
            alpha = 1.0 - norm_position
        else:
            alpha = norm_position

        # Handle shapes for blending
        if len(control_action.shape) > 1:
            alpha = alpha.reshape(-1, 1)

        # Blend: alpha * evade + (1-alpha) * control
        final_action = alpha * evade_action + (1.0 - alpha) * control_action
        
        return final_action, alpha
    
class FatigueBlendingGating(BaseGatingStrategy):
    """
    Blends actions but introduces 'fatigue' to the evade authority. 
    If evade is active for too long, control is gradually handed back.
    """
    def __init__(self, controlModel, evadeModel, controlKeys, evadeKeys, min_val, max_val, metric_extractor, inverted=True, fatigue_rate=0.01, recovery_rate=0.05, max_fatigue=0.8):
        super().__init__(controlModel, evadeModel, controlKeys, evadeKeys)
        self.min_val = min_val
        self.max_val = max_val
        self.metric_extractor = metric_extractor
        self.inverted = inverted
        
        self.fatigue_rate = fatigue_rate
        self.recovery_rate = recovery_rate
        self.max_fatigue = max_fatigue 
        
        self.current_fatigue = None

    def predict(self, obs, deterministic=True):
        control_action, evade_action = self._get_sub_actions(obs, deterministic)
        metric = self.metric_extractor(obs)
        
        clipped_metric = np.clip(metric, self.min_val, self.max_val)
        norm_position = (clipped_metric - self.min_val) / (self.max_val - self.min_val)
        
        base_alpha = 1.0 - norm_position if self.inverted else norm_position
        #print(f"self.current_fatigue: {self.current_fatigue}, base_alpha: {base_alpha}")
        # Initialize fatigue on first step
        if self.current_fatigue is None:
            self.current_fatigue = np.zeros_like(base_alpha) if not np.isscalar(base_alpha) else 0.0

        # Update fatigue: increase if evading, recover if safe
        if np.isscalar(base_alpha):
            if base_alpha > 0.1: 
                self.current_fatigue = min(self.max_fatigue, self.current_fatigue + self.fatigue_rate)
            else:
                self.current_fatigue = max(0.0, self.current_fatigue - self.recovery_rate)
        else:
            evading_mask = base_alpha > 0.1
            increased_fatigue = np.clip(self.current_fatigue + self.fatigue_rate, 0.0, self.max_fatigue)
            decreased_fatigue = np.clip(self.current_fatigue - self.recovery_rate, 0.0, self.max_fatigue)
            self.current_fatigue = np.where(evading_mask, increased_fatigue, decreased_fatigue)

        # Apply fatigue penalty to the base alpha
        effective_alpha = base_alpha * (1.0 - self.current_fatigue)

        # Handle shapes for blending
        alpha_reshaped = effective_alpha
        if len(control_action.shape) > 1 and not np.isscalar(alpha_reshaped):
            alpha_reshaped = alpha_reshaped.reshape(-1, 1)

        final_action = alpha_reshaped * evade_action + (1.0 - alpha_reshaped) * control_action
        
        return final_action, effective_alpha
    
class PredictiveShieldingGating(BaseGatingStrategy):
    """
    Uses a predictive model to forecast future states and determine gating.
    (Placeholder for future implementation)
    """
    def __init__(self, controlModel, evadeModel, controlKeys, evadeKeys, number_of_future_steps, number_intrusor_aircraft=5, intrusion_distance_nm=5, alpha_update_interval=1):
        super().__init__(controlModel, evadeModel, controlKeys, evadeKeys)
        self.number_of_future_steps = number_of_future_steps
        self.number_intrusor_aircraft = number_intrusor_aircraft
        self.intrusion_distance_nm = intrusion_distance_nm
        self.alpha_update_interval = alpha_update_interval  # How often to update alpha (in prediction calls)
        self._prediction_counter = 0
        self._last_alpha = 0.0
    
    def _save_traffic_state(self):
        return {
            # --- Basic Physics ---
            "lat": np.copy(bs.traf.lat),
            "lon": np.copy(bs.traf.lon),
            "hdg": np.copy(bs.traf.hdg),
            "alt": np.copy(bs.traf.alt),
            "tas": np.copy(bs.traf.tas),
            "cas": np.copy(bs.traf.cas),
            "gs": np.copy(bs.traf.gs),
            "trk": np.copy(bs.traf.trk),
            "vs": np.copy(bs.traf.vs),
            "sim_time": bs.sim.simt,

            # --- Kinematics (Hidden State) ---
            "ax": np.copy(bs.traf.ax),           # Current acceleration
            # CHANGE HERE: Use ap.turnphi instead of bank
            "turnphi": np.copy(bs.traf.ap.turnphi), # Current bank angle

            # --- Intermediate Guidance (The 'Switch' variables) ---
            "aporasas_tas": np.copy(bs.traf.aporasas.tas),
            "aporasas_alt": np.copy(bs.traf.aporasas.alt),
            "aporasas_vs":  np.copy(bs.traf.aporasas.vs),
            "aporasas_hdg": np.copy(bs.traf.aporasas.hdg),

            # --- Autopilot Intent ---
            "selspd": np.copy(bs.traf.selspd),
            "swlnav": np.copy(bs.traf.swlnav),
            "swvnav": np.copy(bs.traf.swvnav)
        }

    def _restore_traffic_state(self, state):
        # --- Restore Basic Physics ---
        bs.traf.lat[:] = state["lat"]
        bs.traf.lon[:] = state["lon"]
        bs.traf.hdg[:] = state["hdg"]
        bs.traf.alt[:] = state["alt"]
        bs.traf.tas[:] = state["tas"]
        bs.traf.cas[:] = state["cas"]
        bs.traf.gs[:]  = state["gs"]
        bs.traf.trk[:] = state["trk"]
        bs.traf.vs[:]  = state["vs"]
        bs.sim.simt    = state["sim_time"]

        # --- Restore Kinematics ---
        bs.traf.ax[:] = state["ax"]
        # CHANGE HERE: Restore to ap.turnphi
        bs.traf.ap.turnphi[:] = state["turnphi"]

        # --- Restore Guidance ---
        bs.traf.aporasas.tas[:] = state["aporasas_tas"]
        bs.traf.aporasas.alt[:] = state["aporasas_alt"]
        bs.traf.aporasas.vs[:]  = state["aporasas_vs"]
        bs.traf.aporasas.hdg[:] = state["aporasas_hdg"]

        # --- Restore Autopilot Intent ---
        bs.traf.selspd[:] = state["selspd"]
        bs.traf.swlnav[:] = state["swlnav"]
        bs.traf.swvnav[:] = state["swvnav"]
        
    def _check_for_conflict(self):
        ac_idx = bs.traf.id2idx('KL001')
        
        for i in range(self.number_intrusor_aircraft):
            int_idx = i+1
            _, int_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], bs.traf.lat[int_idx], bs.traf.lon[int_idx])
            if int_dis <= self.intrusion_distance_nm:
                return True
        return False
            
        
    
    def _predict_future_state(self):
        previous_state = self._save_traffic_state()
        steps_without_conflict = 0
        for _ in range(self.number_of_future_steps):
            bs.sim.step()
            if not self._check_for_conflict():
                steps_without_conflict += 1
            else:
                break
        
        self._restore_traffic_state(previous_state)
        #returns alpha evade
        return 1 - steps_without_conflict / self.number_of_future_steps

    def predict(self, obs, deterministic=True):
        control_action, evade_action = self._get_sub_actions(obs, deterministic)
        
        # Only update alpha every alpha_update_interval calls
        if self._prediction_counter % self.alpha_update_interval == 0:
            alpha = self._predict_future_state()
            self._last_alpha = alpha
        else:
            alpha = self._last_alpha
        self._prediction_counter += 1
        # Blend actions based on predicted alpha
        if len(control_action.shape) > 1:
            alpha = np.array(alpha).reshape(-1, 1)
        final_action = alpha * evade_action + (1.0 - alpha) * control_action
        return final_action, alpha
        