import gymnasium as gym
import numpy as np
import pygame
from bluesky_gym.envs.plan_waypoint_env import D_HEADING,ACTION_FREQUENCY,DISTANCE_MARGIN
import bluesky as bs
from bluesky_gym.wrappers.xrlMethods.state.general_actionHeatmap import ActionHeatmapV1Wrapper
from bluesky_gym.envs.common.screen_dummy import ScreenDummy
import bluesky_gym.envs.common.functions as fn
import os
import imageio

import time


# This wrapper creates saliency maps from the current observation
#class SaliencyMapV1Wrapper(gym.ObservationWrapper):





class ActionHeatmapWrapper(ActionHeatmapV1Wrapper):
    
    def __init__(self, env,debug=False, model=None, grid_size=5, grid_spacing_km=10, export_gifs_path=None, fps=5, **kwargs):
        super().__init__(env,debug,grid_size,grid_spacing_km,export_gifs_path, fps, model)
        self.max_distance = 200  # width of screen in km
        self.d_heading = D_HEADING
        


    def lat_lon_to_screen_coordinates(self, lat, lon, *args, **kwargs):
        """
        Converts latitude and longitude to screen coordinates based on the current aircraft position and heading.

        Args:
            lat (float): Latitude of the point to convert.
            lon (float): Longitude of the point to convert."""
            
        # its done reletaive to the center aircraft
        ac_idx = bs.traf.id2idx('KL001')
        
        int_qdr, int_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx],lat, lon)

        x_pos = (self.unwrapped.window_width/2)+(np.sin(np.deg2rad(int_qdr))*(int_dis )/self.max_distance)*self.unwrapped.window_width
        y_pos = (self.unwrapped.window_height/2)-(np.cos(np.deg2rad(int_qdr))*(int_dis )/self.max_distance)*self.unwrapped.window_height
        return int(x_pos), int(y_pos)

    def reset(self, seed=None, options=None):     
        obs,inf = super().reset(seed=seed, options=options)  
        self.episode_counter += 1
        self.step_counter = 0
        
        if self.export_gifs_path is not None:
            # create folder inside frames for this episode
            self.episode_frames_path = os.path.join(self.frames_path, f"episode_{self.episode_counter}")
            os.makedirs(self.episode_frames_path, exist_ok=True)
        if self.unwrapped.render_mode == "human":
            self._render_frame()

        return obs,inf


    def step(self, action):
        
        self.step_counter += 1

        self.unwrapped._get_action(action)
        self.frame_saved = False  # reset frame saved for this step
        action_frequency = ACTION_FREQUENCY
        for i in range(action_frequency):
            bs.sim.step()
            if self.render_mode == "human":
                observation = self.unwrapped._get_obs()
                self._render_frame()

        observation = self.unwrapped._get_obs()
        reward, terminated = self.unwrapped._get_reward()

        info = self.unwrapped._get_info()

        # bluesky reset?? bs.sim.reset()
        if terminated:
            for acid in bs.traf.id:
                idx = bs.traf.id2idx(acid)
                bs.traf.delete(idx)
            if self.export_gifs_path is not None:
                # export gif from saved frames
                gif_filename = os.path.join(self.gifs_path, f"episode_{self.episode_counter}.gif")
                images = [imageio.imread(os.path.join(self.episode_frames_path, f"frame_{step}.png")) for step in range(1, self.step_counter + 1)]
                imageio.mimsave(gif_filename, images, fps=self.fps)

        return observation, reward, terminated, False, info
    
    
    def _render_frame(self):
        self._pre_render()

        max_distance = 200 # width of screen in km

        canvas = pygame.Surface(self.unwrapped.window_size)
        canvas.fill((135,206,235))
        
        observation_grid = self._compute_action_heatmap()
        self._draw_action_heatmap(canvas,observation_grid)
        

        # draw ownship
        ac_idx = bs.traf.id2idx('KL001')
        ac_length = 8
        heading_end_x = ((np.sin(np.deg2rad(bs.traf.hdg[ac_idx])) * ac_length)/max_distance)*self.unwrapped.window_width
        heading_end_y = ((np.cos(np.deg2rad(bs.traf.hdg[ac_idx])) * ac_length)/max_distance)*self.unwrapped.window_width

        pygame.draw.line(canvas,
            (0,0,0),
            (self.unwrapped.window_width/2,self.unwrapped.window_height/2),
            ((self.unwrapped.window_width/2)+heading_end_x,(self.unwrapped.window_height/2)-heading_end_y),
            width = 4
        )

        # draw heading line
        heading_length = 50
        heading_end_x = ((np.sin(np.deg2rad(bs.traf.hdg[ac_idx])) * heading_length)/max_distance)*self.unwrapped.window_width
        heading_end_y = ((np.cos(np.deg2rad(bs.traf.hdg[ac_idx])) * heading_length)/max_distance)*self.unwrapped.window_width

        pygame.draw.line(canvas,
            (0,0,0),
            (self.unwrapped.window_width/2,self.unwrapped.window_height/2),
            ((self.unwrapped.window_width/2)+heading_end_x,(self.unwrapped.window_height/2)-heading_end_y),
            width = 1
        )

        # draw target waypoint
        for qdr, dis, reach in zip(self.unwrapped.wpt_qdr, self.unwrapped.wpt_dis, self.unwrapped.wpt_reach):

            circle_x = ((np.sin(np.deg2rad(qdr)) * dis)/max_distance)*self.unwrapped.window_width
            circle_y = ((np.cos(np.deg2rad(qdr)) * dis)/max_distance)*self.unwrapped.window_width

            if reach:
                color = (155,155,155)
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

        self._post_render(canvas)