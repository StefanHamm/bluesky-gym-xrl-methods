from bluesky_gym.utils.constants import HEADING_LENGTH_IN_SECONDS
import gymnasium as gym
import numpy as np
import pygame
from bluesky_gym.envs.plan_waypoint_env import D_HEADING,ACTION_FREQUENCY,DISTANCE_MARGIN
from bluesky_gym.wrappers.xrlMethods.state.general_saliency import SaliencyMapV1Wrapper
import bluesky as bs
#
import bluesky_gym.envs.common.functions as fn
import os
import copy
import imageio
from bluesky_gym.utils.constants import NM2KM

class SaliencyPlanWaypoint(SaliencyMapV1Wrapper):
    def __init__(self, env, safe_vals=None, debug=False, export_gifs_path=None, fps=5, color_mode="clipped", plot_action_path=False, model=None):
        """
        Initialize the wrapper.

        Args:
            env: The Gym environment to wrap.
            safe_vals (dict, optional): Initial safe values for debugging and visualization.
            debug (bool, optional): If True, enables debug mode for additional visualization and logging.
            export_gifs_path (str, optional): Directory path to export GIFs of episodes. If None, GIFs are not saved.
            fps (int, optional): Frames per second for GIF export and rendering. Default is 5.
            color_mode (str, optional): Color mode for saliency visualization. Use "quantitized", "clipped", or "scaled".
            plot_action_path (bool, optional): If True, plots the action path taken by the agent.
            plot_safe_path (bool, optional): If True, plots the safe action path based on safe values.
        """
        super().__init__(env, safe_vals, debug, export_gifs_path, fps, color_mode, model)
        self.plot_action_path = plot_action_path
        self.action_frequency = ACTION_FREQUENCY #needs to be set since its used inside the general_saliency wrapper but is a global variable in all envs
        self.distance_margin = DISTANCE_MARGIN # same for this one
        self.d_hdg = D_HEADING
        self.max_distance = 200  # width of screen in km
        
    def lat_lon_to_screen_coordinates(self, lat, lon, *args, **kwargs):
        ac_idx = bs.traf.id2idx('KL001')
        qdr, dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], lat, lon)
        
        # Convert distance to KM
        dis_km = dis * NM2KM
        
        # Calculate offsets (scale based on max_distance in KM)
        x_offset = ((np.sin(np.deg2rad(qdr)) * dis_km)/self.max_distance) * self.unwrapped.window_width
        y_offset = ((np.cos(np.deg2rad(qdr)) * dis_km)/self.max_distance) * self.unwrapped.window_width
        
        # Convert to absolute screen coordinates
        # X: center + offset
        # Y: center - offset (Y is inverted in screen coords)
        x_pos = (self.unwrapped.window_width / 2) + x_offset
        y_pos = (self.unwrapped.window_height / 2) - y_offset
        
        return x_pos, y_pos
            
    def reset(self, seed=None, options=None):
        obs,inf = super().reset(seed=seed)
        self.episode_counter += 1
        self.step_counter = 0
        
        if self.export_gifs_path is not None:
            # create folder inside frames for this episode
            self.episode_frames_path = os.path.join(self.frames_path, f"episode_{self.episode_counter}")
            os.makedirs(self.episode_frames_path, exist_ok=True)
    
        
        if self.plot_action_path and self.model is not None:
            prev_reach = self.unwrapped.wpt_reach.copy()
            self._calculate_projected_path(safe=False,has_waypoints=False)
            self.unwrapped.wpt_reach = prev_reach  # reset reach to original after calculation

        if self.render_mode == "human":
            self._render_frame()

        return obs,inf
    
    def step(self, action, shap_values=None,examplePlane = None):
        
        
        self.unwrapped._get_action(action)
        self.last_action = action  # Store the last action
        
        self.step_counter += 1
        self.frame_saved = False  # Reset frame saved flag for this step

        action_frequency = ACTION_FREQUENCY
        for i in range(action_frequency):
            # Moved rendering before the simulation step only saving the first frame, so  the frame matches the observation/action
            if self.render_mode == "human":
                observation = self.unwrapped._get_obs()
                self._render_frame(shap_values=shap_values)
            
            bs.sim.step()
            

        observation = self.unwrapped._get_obs()
        reward, terminated = self.unwrapped._get_reward()

        info = self.unwrapped._get_info()

        if terminated:
            self.export_episode_gif()

        return observation, reward, terminated, False, info
    
    def _render_frame(self, shap_values=None):
        self._pre_render()
        max_distance = 200 # width of screen in km
        canvas = pygame.Surface(self.unwrapped.window_size)
        canvas.fill((135,206,235))
        if self.plot_action_path and self.model is not None:
            self._draw_path(canvas, (255,0,0), self.path_coordinates, True)
        # draw ownship
        ac_idx = bs.traf.id2idx('KL001')
        ac_length = 8
        ac_spd = bs.traf.cas[ac_idx]  # [m/s]
        px_per_km = self.unwrapped.window_width / max_distance
        # Ownship body
        heading_end_x = np.sin(np.deg2rad(bs.traf.hdg[ac_idx])) * ac_length * px_per_km
        heading_end_y = np.cos(np.deg2rad(bs.traf.hdg[ac_idx])) * ac_length * px_per_km
        pygame.draw.line(canvas,
            (0,0,0),
            (self.unwrapped.window_width/2, self.unwrapped.window_height/2),
            (self.unwrapped.window_width/2 + heading_end_x, self.unwrapped.window_height/2 - heading_end_y),
            width=4
        )
        # draw heading line (projected position after HEADING_LENGTH_IN_SECONDS)
        heading_length_km = (ac_spd * HEADING_LENGTH_IN_SECONDS) / 1000.0
        heading_end_x = np.sin(np.deg2rad(bs.traf.hdg[ac_idx])) * heading_length_km * px_per_km
        heading_end_y = np.cos(np.deg2rad(bs.traf.hdg[ac_idx])) * heading_length_km * px_per_km
        pygame.draw.line(canvas,
            (0,0,0),
            (self.unwrapped.window_width/2, self.unwrapped.window_height/2),
            (self.unwrapped.window_width/2 + heading_end_x, self.unwrapped.window_height/2 - heading_end_y),
            width=1
        )

        # draw target waypoint
        for i,(qdr, dis, reach) in enumerate(zip(self.unwrapped.wpt_qdr, self.unwrapped.wpt_dis, self.unwrapped.wpt_reach)):

            circle_x = ((np.sin(np.deg2rad(qdr)) * dis)/max_distance)*self.unwrapped.window_width
            circle_y = ((np.cos(np.deg2rad(qdr)) * dis)/max_distance)*self.unwrapped.window_width

            if reach:
                color = (155,155,155)
            else:
                if shap_values:
                    color = self._get_saliency_color(shap_values.values[0][i],np.max(np.abs(shap_values.values)),shap_values.base_values[0][0])
                else:
                    color = (255,255,255)
            
            pygame.draw.circle(
                canvas, 
                color,
                ((self.unwrapped.window_width/2)+circle_x,(self.unwrapped.window_height/2)-circle_y),
                radius = 4,
                width = 0
            )
            
            pygame.draw.circle(
                canvas, 
                color,
                ((self.unwrapped.window_width/2)+circle_x,(self.unwrapped.window_height/2)-circle_y),
                radius = (DISTANCE_MARGIN/max_distance)*self.unwrapped.window_width,
                width = 2
            )

            # draw a small number indicating intruder index
            index_text = self.font.render(str(i+1), True, (0, 0, 0))
            canvas.blit(index_text, ((self.unwrapped.window_width/2)+circle_x + 5, (self.unwrapped.window_height/2)-circle_y + 5))

         # Draw legend for SHAP influence
        legend_x = 30
        legend_y = self.unwrapped.window_size[1] - 80
        legend_width = 200
        legend_height = 20

      
            
        if shap_values is not None:
            shap_sums = [float(np.sum(shap_values.values))]
            if self.DEBUG:
                
                action_taken = [float(self.last_action)]
                baseline_value = shap_values.base_values[0]
                self._draw_debug_menue(canvas,legend_x,legend_y,action_taken,shap_sums,baseline_value)

            #draw action bar for turn influence
        
            self._draw_shap_bar(canvas,shap_sums[0],legend_x,legend_y,legend_width,legend_height,"horizontal","Right","Left","Overall Turn Influence")

        if not self.frame_saved and self.export_gifs_path is not None and shap_values is not None:
            self._create_shap_row(shap_values.values[0])

        self._post_render(canvas=canvas)