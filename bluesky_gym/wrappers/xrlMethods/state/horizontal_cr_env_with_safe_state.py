

import gymnasium as gym
import numpy as np
import random
from bluesky_gym.envs.horizontal_cr_env import NUM_INTRUDERS, INTRUSION_DISTANCE, WAYPOINT_DISTANCE_MAX, AC_SPD, NM2KM,INTRUSION_PENALTY
import bluesky as bs
import bluesky_gym.envs.common.functions as fn

SAFE_VALS = {
            "dist": 10.0,
            "cos": -1.0,  # Behind
            "sin": 0.0,
            "dx": 0.0,    # Flying away
            "dy": -1.0
        }


class SafeObservationWrapper(gym.Wrapper):
    def __init__(self, env, probability=0.1, safe_intruder_probability=0.5):
        super().__init__(env)
        self.safe_episode = False
        self.safe_intruder_probability = safe_intruder_probability
        self.safe_intruder_indices = []
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        bs.traf.reset()

        self.unwrapped.total_reward = 0
        self.unwrapped.total_intrusions = 0
        self.unwrapped.average_drift = np.array([])

        bs.traf.cre('KL001',actype="A320",acspd=AC_SPD)

        self.unwrapped._generate_conflicts()
        self.unwrapped._generate_waypoint()
        observation = self.unwrapped._get_obs()
        info = self.unwrapped._get_info()

        if self.unwrapped.render_mode == "human":
            self.unwrapped._render_frame()
        
        self.safe_episode = False
        self.safe_intruder_indices = []
        randval = random.random()
        if randval < self.safe_episode:
            self.safe_episode = True
            
        for i in range(NUM_INTRUDERS):
            intruder_randval = random.random()
            if intruder_randval < self.safe_intruder_probability:
                self.safe_intruder_indices.append(i)
                

        return observation, info
    
    def _check_intrusion(self):
        ac_idx = bs.traf.id2idx('KL001')
        reward = 0
        for i in range(NUM_INTRUDERS):
            int_idx = i+1
            _, int_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], bs.traf.lat[int_idx], bs.traf.lon[int_idx])
            if i in self.safe_intruder_indices:
                continue
            if int_dis < INTRUSION_DISTANCE:
                self.total_intrusions += 1
                reward += INTRUSION_PENALTY

    def _get_reward(self):

        # Always return done as false, as this is a non-ending scenario with 
        # new waypoints spawning continously

        reach_reward = self.unwrapped._check_waypoint()
        drift_reward = self.unwrapped._check_drift()
        if self.safe_episode:
            intrusion_reward = 0.0
        else:
            intrusion_reward = self._check_intrusion()

        total_reward = reach_reward + drift_reward + intrusion_reward
        self.unwrapped.total_reward += total_reward

        if 0 in self.unwrapped.wpt_reach:
            return total_reward, 0
        else:
            return total_reward, 1
        
    def _get_obs(self):
        ac_idx = bs.traf.id2idx('KL001')

        self.unwrapped.intruder_distance = []
        self.unwrapped.cos_bearing = []
        self.unwrapped.sin_bearing = []
        self.unwrapped.x_difference_speed = []
        self.unwrapped.y_difference_speed = []

        self.unwrapped.waypoint_distance = []
        self.unwrapped.wpt_qdr = []
        self.unwrapped.cos_drift = []
        self.unwrapped.sin_drift = []
        self.unwrapped.drift = []

        self.unwrapped.ac_hdg = bs.traf.hdg[ac_idx]

        for i in range(NUM_INTRUDERS):
            int_idx = i+1
            int_qdr, int_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], bs.traf.lat[int_idx], bs.traf.lon[int_idx])
        
            self.unwrapped.intruder_distance.append(int_dis * NM2KM)

            bearing = self.unwrapped.ac_hdg - int_qdr
            bearing = fn.bound_angle_positive_negative_180(bearing)

            self.unwrapped.cos_bearing.append(np.cos(np.deg2rad(bearing)))
            self.unwrapped.sin_bearing.append(np.sin(np.deg2rad(bearing)))

            heading_difference = bs.traf.hdg[ac_idx] - bs.traf.hdg[int_idx]
            x_dif = - np.cos(np.deg2rad(heading_difference)) * bs.traf.gs[int_idx]
            y_dif = bs.traf.gs[ac_idx] - np.sin(np.deg2rad(heading_difference)) * bs.traf.gs[int_idx]

            self.unwrapped.x_difference_speed.append(x_dif)
            self.unwrapped.y_difference_speed.append(y_dif)


        for lat, lon in zip(self.unwrapped.wpt_lat, self.unwrapped.wpt_lon):
            
            self.unwrapped.ac_hdg = bs.traf.hdg[ac_idx]
            wpt_qdr, wpt_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], lat, lon)
        
            self.unwrapped.waypoint_distance.append(wpt_dis * NM2KM)
            self.unwrapped.wpt_qdr.append(wpt_qdr)

            drift = self.unwrapped.ac_hdg - wpt_qdr
            drift = fn.bound_angle_positive_negative_180(drift)

            self.unwrapped.drift.append(drift)
            self.unwrapped.cos_drift.append(np.cos(np.deg2rad(drift)))
            self.unwrapped.sin_drift.append(np.sin(np.deg2rad(drift)))

        observation = {
                "intruder_distance": np.array(self.unwrapped.intruder_distance)/WAYPOINT_DISTANCE_MAX,
                "cos_difference_pos": np.array(self.unwrapped.cos_bearing),
                "sin_difference_pos": np.array(self.unwrapped.sin_bearing),
                "x_difference_speed": np.array(self.unwrapped.x_difference_speed)/AC_SPD,
                "y_difference_speed": np.array(self.unwrapped.y_difference_speed)/AC_SPD,
                "waypoint_distance": np.array(self.unwrapped.waypoint_distance)/WAYPOINT_DISTANCE_MAX,
                "cos_drift": np.array(self.unwrapped.cos_drift),
                "sin_drift": np.array(self.unwrapped.sin_drift)
            }
        if self.safe_episode:
            for i in self.safe_intruder_indices: 
                observation["intruder_distance"][i] = SAFE_VALS["dist"]/WAYPOINT_DISTANCE_MAX
                observation["cos_difference_pos"][i] = SAFE_VALS["cos"]
                observation["sin_difference_pos"][i] = SAFE_VALS["sin"]
                observation["x_difference_speed"][i] = SAFE_VALS["dx"]/AC_SPD
                observation["y_difference_speed"][i] = SAFE_VALS["dy"]/AC_SPD
        
        return observation