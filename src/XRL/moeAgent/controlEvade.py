import numpy as np
from abc import ABC, abstractmethod
from stable_baselines3 import SAC,PPO,TD3,DDPG,A2C

import numpy as np
from abc import ABC, abstractmethod
from stable_baselines3 import SAC,PPO,TD3,DDPG,A2C

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
        print(f"Control Action: {control_action}, Evade Action: {evade_action}")
        
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