import gymnasium as gym
import numpy as np
import pygame
from bluesky_gym.envs.sector_cr_env import AC_DENSITY_MU, AC_DENSITY_SIGMA, AC_DENSITY_RANGE, NUM_AC_STATE, ACTION_FREQUENCY, INTRUSION_DISTANCE, NM2KM, D_HEADING, AC_SPD,ACTOR,CENTER
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
    
    def __init__(self, env,debug=False, model=None, grid_size=5, grid_spacing_km=10, export_gifs_path=None, fps=5, plot_action_path=False, **kwargs):
        super().__init__(env,debug,grid_size,grid_spacing_km,export_gifs_path, fps, model)
        self.max_distance = 200  # width of screen in km
        self.d_heading = D_HEADING
        self.px_per_km = None
        self.action_frequency = ACTION_FREQUENCY
        self.plot_action_path = plot_action_path
        
    
    def lat_lon_to_screen_coordinates(self, lat, lon, *args, **kwargs):
        """
        Converts latitude and longitude to screen coordinates based on the current aircraft position and heading.

        Args:
            lat (float): Latitude of the point to convert.
            lon (float): Longitude of the point to convert."""
            
        # its done reletaive to the center aircraft
        ac_idx = bs.traf.id2idx('KL001')
        
        int_qdr, int_dis = bs.tools.geo.kwikqdrdist(CENTER[0],CENTER[1],lat, lon)

        x_pos = (self.unwrapped.window_width/2)+(np.sin(np.deg2rad(int_qdr))*(int_dis * NM2KM)*self.px_per_km)
        y_pos = (self.unwrapped.window_height/2)-(np.cos(np.deg2rad(int_qdr))*(int_dis * NM2KM)*self.px_per_km)
        return int(x_pos), int(y_pos)
    
    def reset(self, seed=None, options=None):       
        self.episode_counter += 1
        self.step_counter = 0
        
        if self.export_gifs_path is not None:
            # create folder inside frames for this episode
            self.episode_frames_path = os.path.join(self.frames_path, f"episode_{self.episode_counter}")
            os.makedirs(self.episode_frames_path, exist_ok=True)
        
        obs, info = super().reset(seed=seed, options=options)

        if self.plot_action_path and self.model is not None:
            self._calculate_projected_path(safe=False,has_waypoints=False)

        return obs, info

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
        reward = self.unwrapped._get_reward()
        info = self.unwrapped._get_info()

        # truncate instead of terminate to avoid aircraft learning to exit sector fast
        truncate = self.unwrapped._check_inside_airspace()

        # bluesky reset?? bs.sim.reset()
        if truncate:
            for acid in bs.traf.id:
                idx = bs.traf.id2idx(acid)
                bs.traf.delete(idx)
            if self.export_gifs_path is not None:
                # export gif from saved frames
                gif_filename = os.path.join(self.gifs_path, f"episode_{self.episode_counter}.gif")
                images = [imageio.imread(os.path.join(self.episode_frames_path, f"frame_{step}.png")) for step in range(1, self.step_counter + 1)]
                imageio.mimsave(gif_filename, images, fps=self.fps)

        return observation, reward, False, truncate, info
    
    
    def _render_frame(self):
        self._pre_render()
        
        ac_idx = bs.traf.id2idx('KL001')
        max_distance = max(np.linalg.norm(point1 - point2) for point1 in self.unwrapped.poly_points for point2 in self.unwrapped.poly_points)*NM2KM

        px_per_km = self.unwrapped.window_width/max_distance
        self.px_per_km = px_per_km
        canvas = pygame.Surface(self.unwrapped.window_size)
        canvas.fill((135,206,235))
        
        if self.plot_action_path and self.model is not None:
             self._draw_path(canvas,(255,0,0),self.path_coordinates,True)
        
        observation_grid = self._compute_action_heatmap()
        self._draw_action_heatmap(canvas,observation_grid)
        
        # Draw airspace
        airspace_color = (255, 0, 0)
        coords = [((self.unwrapped.window_width/2)+point[1]*NM2KM*px_per_km, (self.unwrapped.window_height/2)-point[0]*NM2KM*px_per_km) for point in self.unwrapped.poly_points]
        pygame.draw.polygon(canvas, airspace_color, coords, width=2)
        
        
        

        # Draw ownship
        ac_idx = bs.traf.id2idx(ACTOR)
        ac_length = 10
        ac_hdg = bs.traf.hdg[ac_idx]
        heading_end_x = np.sin(np.deg2rad(ac_hdg)) * ac_length
        heading_end_y = np.cos(np.deg2rad(ac_hdg)) * ac_length
        ac_qdr, ac_dis = bs.tools.geo.kwikqdrdist(CENTER[0], CENTER[1], bs.traf.lat[ac_idx], bs.traf.lon[ac_idx])

        x_pos = (self.unwrapped.window_width/2)+(np.sin(np.deg2rad(ac_qdr))*(ac_dis * NM2KM)*px_per_km)
        y_pos = (self.unwrapped.window_height/2)-(np.cos(np.deg2rad(ac_qdr))*(ac_dis * NM2KM)*px_per_km)

        pygame.draw.line(canvas,
            (0,0,0),
            (x_pos,y_pos),
            ((x_pos)+heading_end_x,(y_pos)-heading_end_y),
            width = 4
        )

        # Draw heading line
        heading_length = 20
        heading_end_x = np.sin(np.deg2rad(ac_hdg)) * heading_length
        heading_end_y = np.cos(np.deg2rad(ac_hdg)) * heading_length

        pygame.draw.line(canvas,
                (0,0,0),
                (x_pos,y_pos),
                ((x_pos)+heading_end_x,(y_pos)-heading_end_y),
                width = 1
        )

        # draw intruders
        ac_length = 3

        for i in range(self.unwrapped.num_ac-1):
            int_idx = i+1
            int_hdg = bs.traf.hdg[int_idx]
            heading_end_x = np.cos(np.deg2rad(int_hdg)) * ac_length
            heading_end_y = np.sin(np.deg2rad(int_hdg)) * ac_length

            int_qdr, int_dis = bs.tools.geo.kwikqdrdist(CENTER[0], CENTER[1], bs.traf.lat[int_idx], bs.traf.lon[int_idx])
            separation = bs.tools.geo.kwikdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], bs.traf.lat[int_idx], bs.traf.lon[int_idx])

            # Determine color
            if separation < INTRUSION_DISTANCE:
                color = (220,20,60)
            else: 
                color = (80,80,80)

            x_pos = (self.unwrapped.window_width/2)+(np.sin(np.deg2rad(int_qdr))*(int_dis * NM2KM)*px_per_km)
            y_pos = (self.unwrapped.window_height/2)-(np.cos(np.deg2rad(int_qdr))*(int_dis * NM2KM)*px_per_km)

            pygame.draw.line(canvas,
                color,
                (x_pos,y_pos),
                ((x_pos)+heading_end_x,(y_pos)-heading_end_y),
                width = 4
            )

            # Draw heading line
            heading_length = 20
            heading_end_x = np.sin(np.deg2rad(int_hdg)) * heading_length
            heading_end_y = np.cos(np.deg2rad(int_hdg)) * heading_length

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
                radius = INTRUSION_DISTANCE*NM2KM*px_per_km,
                width = 2
            )

        self._post_render(canvas)