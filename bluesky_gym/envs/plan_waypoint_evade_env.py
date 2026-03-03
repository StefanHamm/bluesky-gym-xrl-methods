import numpy as np
import pygame
import copy
import bluesky as bs
import bluesky_gym.envs.common.functions as fn
from bluesky_gym.utils.constants import HEADING_LENGTH_IN_SECONDS
import gymnasium as gym
from gymnasium import spaces
import os
NM2KM = 1.852

SAFE_DISTANCE = 55 #NM

DISTANCE_MARGIN = 5 # km
WAYPOINT_DISTANCE_MIN = 0
WAYPOINT_DISTANCE_MAX = 75

ACTION_PENALTY = -0.2
SMOOTHNESS_PENALTY = -0.5
ALIVE_PENALTY = -0.05

NUM_WAYPOINTS = 5

REACH_REWARD = 15
AC_SPD = 150

D_HEADING = 45


ACTION_FREQUENCY = 10


from bluesky_gym.envs.free_flight_env import FreeFlightCREnv,INTRUSION_DISTANCE,NUM_INTRUDERS

class PlanWaypointEvadeEnv(FreeFlightCREnv):
    """ 
    Cobines the Flight Env with the waypoint reaching of the PlanWayointEnv
    
    """

    # information regarding the possible rendering modes of the environment
    # for BlueSkyGym probably only implement 1 for now together with None, which is default
    metadata = {"render_modes": ["rgb_array","human"], "render_fps": 120}

    def __init__(self, render_mode=None,workdir=None,training = True,fps=5,export_gifs_path=None,moe_rendering=True):
        super().__init__(render_mode=render_mode, workdir=workdir,fps=fps,export_gifs_path=export_gifs_path)
        self.training = training
        
        
        self.observation_space.spaces.update(
            {
                "waypoint_distance": spaces.Box(-np.inf, np.inf, shape = (NUM_WAYPOINTS,), dtype=np.float64),
                "cos_difference": spaces.Box(-np.inf, np.inf, shape = (NUM_WAYPOINTS,), dtype=np.float64),
                "sin_difference": spaces.Box(-np.inf, np.inf, shape = (NUM_WAYPOINTS,), dtype=np.float64),
                "waypoint_reached": spaces.Box(0, 1, shape = (NUM_WAYPOINTS,), dtype=np.float64),
                "previous_action": spaces.Box(-1, 1, shape=(1,), dtype=np.float64)
            }
        )
        
        if training:
            # two things need to be removed from the observation space
            # heading difference cos and sin
            
            del self.observation_space.spaces["cos_own_heading"]
            del self.observation_space.spaces["sin_own_heading"]
        
        #self.action_space = spaces.Box(-1, 1, shape=(1,), dtype=np.float64)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # initialize bluesky as non-networked simulation node
        if bs.sim is None:
            bs.init(mode='sim', detached=True,workdir=workdir)

        # set correct sim speed
        bs.stack.stack('DT 1;FF')

        # initialize values used for logging -> input in _get_info
        self.total_reward = 0
        self.waypoints_completed = 0

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        self.moe_rendering = moe_rendering
        # initialize waypoints
        self.wpt_lat = []
        self.wpt_lon = []
        self.wpt_reach = []
        self.last_action = np.array([0.0], dtype=np.float64)
        self.evading = 0
        self.respawn_intruder = np.array([False]*NUM_INTRUDERS)
        self.prev_intruder_distance = np.array([np.inf]*NUM_INTRUDERS)
        self.episode_counter = -1
        
        self.episode_frames_path =None
        

    def _generate_conflicts(self, acid = 'KL001'):
        target_idx = bs.traf.id2idx(acid)
        for i in range(NUM_INTRUDERS):
            dpsi = self.np_random.integers(0,360)
            cpa = self.np_random.integers(0,INTRUSION_DISTANCE)
            tlosh = self.np_random.integers(50,400)
            bs.traf.creconfs(acid=f'{i}',actype="A320",targetidx=target_idx,dpsi=dpsi,dcpa=cpa,tlosh=tlosh)

    

    def _get_obs(self):
        """
        Observation consists of distance to the waypoint and heading difference with respect to the waypoint
        in cosine and sine decomposition.

        """
        
        parent_obs = super()._get_obs()
        
        if not self.training:
            # set the heading sin cos to 0 degree since we dont have a specific heading to reach
            parent_obs["sin_own_heading"] = np.array([0], dtype=np.float64)
            parent_obs["cos_own_heading"] = np.array([1], dtype=np.float64)
        else:
            #remove from dict
            del parent_obs["sin_own_heading"]
            del parent_obs["cos_own_heading"]
       
        ac_idx = bs.traf.id2idx('KL001')

        self.wpt_dis = []
        self.wpt_qdr = []
        self.drift = []
        self.wpt_cos = []
        self.wpt_sin = []
        
        for lat, lon in zip(self.wpt_lat, self.wpt_lon):
            
            self.ac_hdg = bs.traf.hdg[ac_idx]
            wpt_qdr, wpt_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], lat, lon)
        
            self.wpt_dis.append(wpt_dis * NM2KM)
            self.wpt_qdr.append(wpt_qdr)

            drift = self.ac_hdg - wpt_qdr
            drift = fn.bound_angle_positive_negative_180(drift)

            self.wpt_cos.append(np.cos(np.deg2rad(drift)))
            self.wpt_sin.append(np.sin(np.deg2rad(drift)))
            self.drift.append(drift)

        observation = {
                "waypoint_distance": (np.array(self.wpt_reach) -1)* -1 * np.clip(np.array(self.wpt_dis)/(2*WAYPOINT_DISTANCE_MAX), 0, 1),
                "cos_difference": (np.array(self.wpt_reach) -1)* -1 * np.array(self.wpt_cos),
                "sin_difference": (np.array(self.wpt_reach) -1)* -1 * np.array(self.wpt_sin),
                "waypoint_reached": np.array(self.wpt_reach),
                "previous_action": np.array(self.last_action)
            }
        
        return observation | parent_obs
    
    def _get_info(self):
        # Here you implement any additional info that you want to return after a step,
        # but that should not be used by the agent for decision making, so used for logging and debugging purposes
        # for now just have 10, because it crashed if I gave none for some reason.
        return {
            "total_reward": self.total_reward,
            "waypoints_completed": self.waypoints_completed
        } | super()._get_info()
        
    def _get_action_penalty(self):
        return np.abs(self.current_action[0])*ACTION_PENALTY
    
    def _get_smoothness_penalty(self):
        return SMOOTHNESS_PENALTY * np.abs(self.current_action[0] - self.last_action[0])
    
    def _get_alive_penalty(self):
        # 1. Get distances to active waypoints
        unreached_distances = [
            d for d, r in zip(self.wpt_dis, self.wpt_reach) if r == 0
        ]

        if not unreached_distances:
            return ALIVE_PENALTY

        # 2. Find closest distance
        closest_dist = min(unreached_distances)
        
        # 3. Normalize by DIAMETER (2x Max Radius)
        # e.g. 150 km
        MAX_VALID_SEPARATION = 2 * WAYPOINT_DISTANCE_MAX
        dist_ratio = closest_dist / MAX_VALID_SEPARATION

        # 4. Apply Logic
        if dist_ratio > 1.0:
            # Agent is further than 150km from the target.
            # This implies it is flying away from the arena.
            # Scale penalty quadratically.
            return ALIVE_PENALTY * (dist_ratio ** 2)
        else:
            # Agent is within a valid traversal distance.
            # Constant penalty to encourage speed, but no extra punishment.
            return ALIVE_PENALTY
    def _get_heading_change_penalty(self):
        # overriden for parent to return 0 since this penalty is not applied anymore
        return 0
    
    def _get_reward(self):

        # Always return done as false, as this is a non-ending scenario with 
        # new waypoints spawning continously
        
        parent_reward, terminated = super()._get_reward()

        reach_reward = self._check_waypoint()
        #action_penalty = self._get_action_penalty() gets already called in parent
        alive_penalty = self._get_alive_penalty()
        smoothness_penalty = self._get_smoothness_penalty()
        
        total_reward = reach_reward + alive_penalty + smoothness_penalty + parent_reward
        self.total_reward += total_reward

        if 0 in self.wpt_reach and not terminated:
            return total_reward, 0
        else:
            return total_reward, 1
        
    def _get_action(self,action):
        self.current_action = action
        # Transform action to the change in heading
        # action = self.np_random.integers(-100,100)/100
        action = self.ac_hdg + action * D_HEADING
        
        bs.stack.stack(f"HDG KL001 {action[0]}")

    def reset(self, seed=None, options=None):
        
        self.episode_counter+=1
        if self.export_gifs_path is not None and self.episode_counter >0:
            # create folder inside frames for this episode
            self.episode_frames_path = os.path.join(self.frames_path, f"episode_{self.episode_counter}")
            os.makedirs(self.episode_frames_path, exist_ok=True)
        
        
        
        

        self.total_reward = 0
        
        self.waypoints_completed = 0
        self.last_action = np.array([0.0], dtype=np.float64)
        self.step_counter=0
        self.respawn_intruder = np.array([False]*NUM_INTRUDERS)
        self.intruder_distance = np.array([np.inf]*NUM_INTRUDERS)
        self.prev_intruder_distance = np.array([np.inf]*NUM_INTRUDERS)

        #bs.traf.cre('KL001',actype="A320",acspd=AC_SPD)
        super().reset(seed=seed)
        self._generate_waypoint()
        observation = self._get_obs()
        info = self._get_info()

        # if self.render_mode == "human":
        #     self._render_frame()

        return observation, info
    
    def step(self, action):
        
        

        self._get_action(action)
        self.last_action = action
        self.step_counter +=1
        
        self.prev_intruder_distance = copy.deepcopy(self.intruder_distance)
        self.frame_saved= False
        action_frequency = ACTION_FREQUENCY
        for i in range(action_frequency):
            bs.sim.step()
            if self.render_mode == "human":
                observation = self._get_obs()
                self._render_frame()

        observation = self._get_obs()
        reward, terminated = self._get_reward()

        info = self._get_info()
        
        self._mark_intruders_for_respawn()
        self._perform_respawns()

        # bluesky reset?? bs.sim.reset()
        if terminated:
            bs.traf.reset()

        return observation, reward, terminated, False, info
    
    def _perform_respawns(self):
        for i in range(NUM_INTRUDERS):
            if self.respawn_intruder[i]:
                #deltete intruder with id "i"
                idx = bs.traf.id2idx(f'{i}')
                if idx >= 0:
                    bs.traf.delete(idx)
                
                # respawn intruder
                target_idx = bs.traf.id2idx('KL001')
                if target_idx >= 0:
                    self._create_single_conflict(i,target_idx)
                
                # Reset distance to inf so next step doesn't immediately flag it again
                self.intruder_distance[i] = np.inf
                self.respawn_intruder[i] = False

    def _mark_intruders_for_respawn(self):
        for i,prev_dist in enumerate(self.prev_intruder_distance):
            if prev_dist< self.intruder_distance[i] and self.intruder_distance[i] >= SAFE_DISTANCE * NM2KM:
                self.respawn_intruder[i] = True
                #break#need to stop after one respawn to avoid index mixup
    

    def _generate_waypoint(self, acid = 'KL001'):
        self.wpt_lat = []
        self.wpt_lon = []
        self.wpt_reach = []
        for i in range(NUM_WAYPOINTS):
            wpt_dis_init = self.np_random.integers(WAYPOINT_DISTANCE_MIN, WAYPOINT_DISTANCE_MAX)
            wpt_hdg_init = self.np_random.integers(0, 359)

            ac_idx = bs.traf.id2idx(acid)

            wpt_lat, wpt_lon = fn.get_point_at_distance(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], wpt_dis_init, wpt_hdg_init)    
            self.wpt_lat.append(wpt_lat)
            self.wpt_lon.append(wpt_lon)
            self.wpt_reach.append(0)

    def _check_waypoint(self):
        reward = 0
        index = 0
        for distance in self.wpt_dis:
            if distance < DISTANCE_MARGIN and self.wpt_reach[index] != 1:
                self.waypoints_completed += 1
                self.wpt_reach[index] = 1
                reward += REACH_REWARD
                index += 1
            else:
                reward += 0
                index += 1
        return reward
    
    def update_evading_status(self,evading):
        self.evading = evading

    def _render_frame(self):
        self._pre_render()

        max_distance = 200 # width of screen in km

        canvas = pygame.Surface(self.window_size)
        canvas.fill((135,206,235))

        # draw ownship
        ac_idx = bs.traf.id2idx('KL001')
        ac_length = 8
        
        heading_end_x = ((np.sin(np.deg2rad(bs.traf.hdg[ac_idx])) * ac_length)/max_distance)*self.window_width
        heading_end_y = ((np.cos(np.deg2rad(bs.traf.hdg[ac_idx])) * ac_length)/max_distance)*self.window_width

        if self.moe_rendering:
            evading_color = 255*self.evading
        else:
            evading_color = 0

        pygame.draw.line(canvas,
            (evading_color,0,0),
            (self.window_width/2,self.window_height/2),
            ((self.window_width/2)+heading_end_x,(self.window_height/2)-heading_end_y),
            width = 4
        )

        # draw heading line
        heading_length = 50
        ac_spd = bs.traf.cas[ac_idx]
        km2px = self.window_width / max_distance
        heading_length_km = HEADING_LENGTH_IN_SECONDS * ac_spd / 1000
        heading_length_px = heading_length_km * km2px
        
        
        heading_end_x = np.sin(np.deg2rad(bs.traf.hdg[ac_idx])) * heading_length_px
        heading_end_y = np.cos(np.deg2rad(bs.traf.hdg[ac_idx])) * heading_length_px

        pygame.draw.line(canvas,
            (evading_color,0,0),
            (self.window_width/2,self.window_height/2),
            ((self.window_width/2)+heading_end_x,(self.window_height/2)-heading_end_y),
            width = 1
        )

        # draw target waypoint
        for qdr, dis, reach in zip(self.wpt_qdr, self.wpt_dis, self.wpt_reach):

            circle_x = ((np.sin(np.deg2rad(qdr)) * dis)/max_distance)*self.window_width
            circle_y = ((np.cos(np.deg2rad(qdr)) * dis)/max_distance)*self.window_width

            if reach:
                color = (155,155,155)
            else:
                color = (255,255,255)

            pygame.draw.circle(
                canvas, 
                color,
                ((self.window_width/2)+circle_x,(self.window_height/2)-circle_y),
                radius = 4,
                width = 0
            )
            
            pygame.draw.circle(
                canvas, 
                color,
                ((self.window_width/2)+circle_x,(self.window_height/2)-circle_y),
                radius = (DISTANCE_MARGIN/max_distance)*self.window_width,
                width = 2
            )
            
            
       

        # draw intruders
        ac_length = 3

        for i in range(NUM_INTRUDERS):
            int_idx = i+1
            int_hdg = bs.traf.hdg[int_idx]
            heading_end_x = ((np.sin(np.deg2rad(int_hdg)) * ac_length)/max_distance)*self.window_width
            heading_end_y = ((np.cos(np.deg2rad(int_hdg)) * ac_length)/max_distance)*self.window_width

            int_qdr, int_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], bs.traf.lat[int_idx], bs.traf.lon[int_idx])

            # determine color
            if int_dis < INTRUSION_DISTANCE:
                color = (220,20,60)
            else: 
                color = (80,80,80)

            x_pos = (self.window_width/2)+(np.sin(np.deg2rad(int_qdr))*(int_dis * NM2KM)/max_distance)*self.window_width
            y_pos = (self.window_height/2)-(np.cos(np.deg2rad(int_qdr))*(int_dis * NM2KM)/max_distance)*self.window_height

            pygame.draw.line(canvas,
                color,
                (x_pos,y_pos),
                ((x_pos)+heading_end_x,(y_pos)-heading_end_y),
                width = 4
            )

            # draw heading line
            
            int_spd = bs.traf.cas[int_idx]  
            heading_length_km = HEADING_LENGTH_IN_SECONDS * int_spd / 1000
            heading_length_px = heading_length_km * km2px
            heading_end_x = np.sin(np.deg2rad(int_hdg)) * heading_length_px
            heading_end_y = np.cos(np.deg2rad(int_hdg)) * heading_length_px


            pygame.draw.line(canvas,
                color,
                (x_pos,y_pos),
                ((x_pos)+heading_end_x,(y_pos)-heading_end_y),
                width = 1
            )

            pygame.draw.circle(
                canvas, 
                color,
                (x_pos,y_pos),
                radius = (INTRUSION_DISTANCE*NM2KM/max_distance)*self.window_width,
                width = 2
            )

        def _draw_gradient_bar( canvas, start_pos, length, thickness, horizontal=True):
            """
            Draws a gradient bar (Blue -> Grey -> Red) on the canvas.
            """
            for i in range(length):
                # Normalize i to [-1, 1] for color calculation
                # value = (i / length) * 2 - 1
                # val = max(-1, min(1, value))
                val = (i / length)
                # Get color
                # if val < 0:
                #     t = -val
                #     color = (int(80 * (1-t)), int(80 * (1-t)), int(80 * (1-t) + 255 * t))
                # else:
                #     t = val
                #     color = (int(80 * (1-t) + 255 * t), int(80 * (1-t)), int(80 * (1-t)))
                color = (int(255 * val),0,0)
                # Draw slice
                if horizontal:
                    # x varies, y is constant block
                    pygame.draw.line(canvas, color, (start_pos[0] + i, start_pos[1]), 
                                                (start_pos[0] + i, start_pos[1] + thickness-3), 1)
                else:
                    # y varies, x is constant block
                    pygame.draw.line(canvas, color, (start_pos[0], start_pos[1] + length - i), 
                                                (start_pos[0] + thickness-3, start_pos[1] + length - i), 1)

            evade_pos = self.evading * (length - 1)
            pygame.draw.line(canvas, (255,255,255), (start_pos[0] + evade_pos, start_pos[1]), 
                                        (start_pos[0] + evade_pos, start_pos[1] + thickness-3), 2)
            pygame.draw.rect(canvas, (0,0,0), (start_pos[0]-1, start_pos[1], length+3, thickness), 2)

            pos_text = f'Evade'
            neg_text = f'Control'
            title_text = f'Gating Metric: {self.evading:.2f}'
            font = pygame.font.SysFont(None, 24)
            
            pos_text = font.render(pos_text, True, (0,0,0))
            neg_text = font.render(neg_text, True, (0,0,0))
            title_text = font.render(title_text, True, (0,0,0))
            # Get dimensions
            pos_w, pos_h = pos_text.get_size()
            neg_w, neg_h = neg_text.get_size()
            title_w, title_h = title_text.get_size()
            
            canvas.blit(neg_text, (start_pos[0] , start_pos[1] + thickness + 5))
            canvas.blit(pos_text, (start_pos[0] + length - pos_w, start_pos[1] + thickness + 5))
            canvas.blit(title_text, (start_pos[0] + (length- title_w)//2, start_pos[1] - 20))
            
        startpos = (10, self.window_height - 40)
        if self.moe_rendering:
            _draw_gradient_bar(canvas, startpos, 200, 20, horizontal=True)

        self._post_render(canvas)
        
    def close(self):
        bs.stack.stack('quit')