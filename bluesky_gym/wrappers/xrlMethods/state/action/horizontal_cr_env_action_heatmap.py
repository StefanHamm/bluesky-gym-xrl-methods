from bluesky_gym.utils.constants import HEADING_LENGTH_IN_SECONDS
import gymnasium as gym
import numpy as np
import pygame
from bluesky_gym.envs.horizontal_cr_env import D_HEADING,ACTION_FREQUENCY,NUM_INTRUDERS,NM2KM,INTRUSION_DISTANCE,DISTANCE_MARGIN,AC_SPD,WAYPOINT_DISTANCE_MAX
import bluesky as bs
from bluesky_gym.wrappers.xrlMethods.state.general_actionHeatmap import ActionHeatmapV1Wrapper
#
import bluesky_gym.envs.common.functions as fn
import os
import imageio

import time


# This wrapper creates saliency maps from the current observation
#class SaliencyMapV1Wrapper(gym.ObservationWrapper):





class ActionHeatmapWrapper(ActionHeatmapV1Wrapper):
    
    def __init__(self, env,debug=False, model=None, grid_size=5, grid_spacing_km=10, export_gifs_path=None, fps=5,point_to_waypoint = True,plot_action_path=False, xrl_rendering=True, **kwargs):
        super().__init__(env,debug,grid_size,grid_spacing_km,export_gifs_path, fps, model, xrl_rendering=xrl_rendering)
        self.max_distance = 200  # width of screen in km
        self.d_heading = D_HEADING      
        self.action_frequency = ACTION_FREQUENCY
        self.distance_margin = DISTANCE_MARGIN
        self.plot_action_path = plot_action_path
        self.point_to_waypoint = point_to_waypoint

    def lat_lon_to_screen_coordinates(self, lat, lon, *args, **kwargs):
        """
        Converts latitude and longitude to screen coordinates based on the current aircraft position and heading.

        Args:
            lat (float): Latitude of the point to convert.
            lon (float): Longitude of the point to convert."""
            
        # its done reletaive to the center aircraft
        ac_idx = bs.traf.id2idx('KL001')
        
        int_qdr, int_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx],lat, lon)

        x_pos = (self.unwrapped.window_width/2)+(np.sin(np.deg2rad(int_qdr))*(int_dis * NM2KM)/self.max_distance)*self.unwrapped.window_width
        y_pos = (self.unwrapped.window_height/2)-(np.cos(np.deg2rad(int_qdr))*(int_dis * NM2KM)/self.max_distance)*self.unwrapped.window_height
        return int(x_pos), int(y_pos)

    def reset(self, seed=None, options=None):     
        obs,inf = super().reset(seed=seed, options=options)  
        self.episode_counter += 1
        self.step_counter = 0

        if self.plot_action_path and self.model is not None:
            self._calculate_projected_path(safe=False,has_waypoints=True)
        
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
        ac_idx = bs.traf.id2idx('KL001')
        canvas = pygame.Surface(self.unwrapped.window_size)
        canvas.fill((135,206,235))
        if self.plot_action_path and self.model is not None:
            self._draw_path(canvas, (255,0,0), self.path_coordinates, True)
        if self.point_to_waypoint:
            observation_grid = self._compute_action_heatmap((self.unwrapped.wpt_lat[0], self.unwrapped.wpt_lon[0]))
        else:
            observation_grid = self._compute_action_heatmap()
            
        if self.xrl_rendering:
            self._draw_action_heatmap(canvas, observation_grid)
        # draw ownship
        ac_length = 8
        ac_spd = bs.traf.cas[ac_idx]  # [m/s]
        px_per_km = self.unwrapped.window_width / self.max_distance
        # Ownship body
        heading_end_x = np.sin(np.deg2rad(bs.traf.hdg[ac_idx])) * ac_length * px_per_km
        heading_end_y = np.cos(np.deg2rad(bs.traf.hdg[ac_idx])) * ac_length * px_per_km
        pygame.draw.line(canvas,
            (0,0,0),
            (self.unwrapped.window_width/2 - heading_end_x/2, self.unwrapped.window_height/2 + heading_end_y/2),
            (self.unwrapped.window_width/2 + heading_end_x/2, self.unwrapped.window_height/2 - heading_end_y/2),
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

        # draw intruders
        ac_length = 3

        for i in range(NUM_INTRUDERS):
            int_idx = i+1
            int_hdg = bs.traf.hdg[int_idx]
            heading_end_x = ((np.sin(np.deg2rad(int_hdg)) * ac_length)/self.max_distance)*self.unwrapped.window_width
            heading_end_y = ((np.cos(np.deg2rad(int_hdg)) * ac_length)/self.max_distance)*self.unwrapped.window_width

            int_qdr, int_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], bs.traf.lat[int_idx], bs.traf.lon[int_idx])

            # determine color
            if int_dis < INTRUSION_DISTANCE:
                color = (220,20,60)
            else: 
                color = (80,80,80)

            x_pos = (self.unwrapped.window_width/2)+(np.sin(np.deg2rad(int_qdr))*(int_dis * NM2KM)/self.max_distance)*self.unwrapped.window_width
            y_pos = (self.unwrapped.window_height/2)-(np.cos(np.deg2rad(int_qdr))*(int_dis * NM2KM)/self.max_distance)*self.unwrapped.window_height

            pygame.draw.line(canvas,
                color,
                (x_pos,y_pos),
                ((x_pos)+heading_end_x,(y_pos)-heading_end_y),
                width = 4
            )

            # draw heading line
            int_idx = bs.traf.id2idx(str(int_idx))
            int_spd = bs.traf.cas[int_idx]
            #print(int_spd,ac_spd)
            heading_length_km = (int_spd * HEADING_LENGTH_IN_SECONDS) / 1000.0
            heading_length_px = heading_length_km * px_per_km
            #print(int_spd,ac_spd)
            heading_end_y = ((np.cos(np.deg2rad(int_hdg)) * heading_length_px))
            heading_end_x = ((np.sin(np.deg2rad(int_hdg)) * heading_length_px))
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
                radius = (INTRUSION_DISTANCE*NM2KM/self.max_distance)*self.unwrapped.window_width,
                width = 2
            )

            # import code
            # code.interact(local=locals())

        # draw target waypoint
        for qdr, dis, reach in zip(self.unwrapped.wpt_qdr, self.unwrapped.waypoint_distance, self.unwrapped.wpt_reach):

            circle_y = ((np.cos(np.deg2rad(qdr)) * dis)/self.max_distance)*self.unwrapped.window_width
            circle_x = ((np.sin(np.deg2rad(qdr)) * dis)/self.max_distance)*self.unwrapped.window_width

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
                radius = (DISTANCE_MARGIN/self.max_distance)*self.unwrapped.window_width,
                width = 2
            )

        self._post_render(canvas)