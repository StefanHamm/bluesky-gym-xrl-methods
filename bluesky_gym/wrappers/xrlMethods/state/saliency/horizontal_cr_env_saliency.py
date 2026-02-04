import gymnasium as gym
import numpy as np
import pygame
from bluesky_gym.envs.horizontal_cr_env import D_HEADING,ACTION_FREQUENCY,NUM_INTRUDERS,NM2KM,INTRUSION_DISTANCE,DISTANCE_MARGIN,AC_SPD,WAYPOINT_DISTANCE_MAX
from bluesky_gym.wrappers.xrlMethods.state.general_saliency import SaliencyMapV1Wrapper
import bluesky as bs
from bluesky_gym.envs.common.screen_dummy import ScreenDummy
import bluesky_gym.envs.common.functions as fn
import os
import copy
import imageio



class SaliencyHorizontalControl(SaliencyMapV1Wrapper):
    
    def __init__(self, env, safe_vals=None, debug=False, export_gifs_path=None, fps=5, color_mode="clipped", plot_action_path=False, plot_safe_path=False, model=None):
        """
        Initialize the SaliencyHorizontalControl wrapper.

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
        self.action_frequency = ACTION_FREQUENCY #needs to be set since its used inside the general_saliency wrapper but is a global variable in all envs
        self.distance_margin = DISTANCE_MARGIN # same for this one
        self.num_intruders = NUM_INTRUDERS
        self.d_hdg = D_HEADING
        self.plot_action_path = plot_action_path
        self.plot_safe_path = plot_safe_path
            
            
    def reset(self, seed=None, options=None):
        obs,inf = super().reset(seed=seed)
        self.episode_counter += 1
        self.step_counter = 0
        
        if self.export_gifs_path is not None:
            # create folder inside frames for this episode
            self.episode_frames_path = os.path.join(self.frames_path, f"episode_{self.episode_counter}")
            os.makedirs(self.episode_frames_path, exist_ok=True)
    
        
        if self.plot_action_path and self.model is not None:
            self._calculate_projected_path(safe=False,has_waypoints=True)
            

        if self.render_mode == "human":
            self._render_frame()

        return obs,inf
    
            
    def step(self, action, shap_values=None,examplePlane = None):
        
        if self.plot_safe_path and self.model is not None and self.safe_vals is not None:
             self._calculate_projected_path(safe=True,has_waypoints=True)
        
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
    
    def lat_lon_to_screen_coordinates (self,lat,lon):
        ac_idx = bs.traf.id2idx('KL001')
        qdr, dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], lat, lon)
        x_pos = (self.unwrapped.window_width/2)+(np.sin(np.deg2rad(qdr))*(dis * NM2KM)/200)*self.unwrapped.window_width
        y_pos = (self.unwrapped.window_height/2)-(np.cos(np.deg2rad(qdr))*(dis * NM2KM)/200)*self.unwrapped.window_height
        return x_pos,y_pos
        
    
    def _render_frame(self,shap_values=None):
        self._pre_render()
                
        ac_idx = bs.traf.id2idx('KL001')        
                  
        max_distance = 200 # width of screen in km

        canvas = pygame.Surface(self.unwrapped.window_size)
        canvas.fill((135,206,235))

        if self.plot_action_path and self.model is not None:
            self._draw_path(canvas,(255,0,0),self.path_coordinates,True)
           
                
        if self.plot_safe_path and self.model is not None and self.safe_vals is not None:
            self._draw_path(canvas,(255,0,255),self.safe_action_path)
                
        # draw ownship
        
        ac_length = 8
        heading_end_x = ((np.sin(np.deg2rad(bs.traf.hdg[ac_idx])) * ac_length)/max_distance)*self.unwrapped.window_width
        heading_end_y = ((np.cos(np.deg2rad(bs.traf.hdg[ac_idx])) * ac_length)/max_distance)*self.unwrapped.window_width

        pygame.draw.line(canvas,
            (0,0,0),
            (self.unwrapped.window_width/2-heading_end_x/2,self.unwrapped.window_height/2+heading_end_y/2),
            ((self.unwrapped.window_width/2)+heading_end_x/2,(self.unwrapped.window_height/2)-heading_end_y/2),
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

        # Plot additionally the intended_heading (baseline heading)
        if shap_values is not None and self.DEBUG:

                
            base_val  = shap_values.base_values[0][0]

            # intended_heading = current_heading + baseline_turn
            intended_heading = bs.traf.hdg[ac_idx] + base_val * D_HEADING
            
            heading_end_x_intend = ((np.sin(np.deg2rad(intended_heading)) * heading_length)/max_distance)*self.unwrapped.window_width
            heading_end_y_intend = ((np.cos(np.deg2rad(intended_heading)) * heading_length)/max_distance)*self.unwrapped.window_width
            
            # Draw baseline/intended heading as a Green line
            pygame.draw.line(canvas,
                (0,255,0),
                (self.unwrapped.window_width/2,self.unwrapped.window_height/2),
                ((self.unwrapped.window_width/2)+heading_end_x_intend,(self.unwrapped.window_height/2)-heading_end_y_intend),
                width = 2
            )

        if self.DEBUG:
            #plot one intrude (now plotted relative to ownship heading)
            color = (0,255,0)

            # compute relative bearing from cos/sin (these encode ac_hdg - qdr)
            rel_bearing_rad = np.arctan2(self.safe_vals["sin_difference_pos"], self.safe_vals["cos_difference_pos"])  # rel = ac_hdg - qdr
            rel_bearing_deg = np.rad2deg(rel_bearing_rad)
            # convert to global bearing from ownship to intruder
            int_qdr = (bs.traf.hdg[ac_idx] - rel_bearing_deg) % 360

            # CORRECT DISTANCE CALCULATION:
            # safe_vals["dist"] is normalized (0-1), so we multiply by MAX to get KM
            dist_km = self.safe_vals["intruder_distance"] * WAYPOINT_DISTANCE_MAX
            
            # Use consistent scale for X and Y to prevent distortion
            screen_scale = self.unwrapped.window_height 

            x_pos = (self.unwrapped.window_width/2) + (np.sin(np.deg2rad(int_qdr)) * dist_km / max_distance) * screen_scale
            y_pos = (self.unwrapped.window_height/2) - (np.cos(np.deg2rad(int_qdr)) * dist_km / max_distance) * screen_scale

            # compute intruder heading: rotate local (dx,dy) by ownship heading
            heading_mag = np.sqrt(self.safe_vals["x_difference_speed"]**2 + self.safe_vals["y_difference_speed"]**2)
            if heading_mag > 1e-8:
                # REVERSE TRANSFORM SPEED DIFFERENCE TO HEADING
                # x_dif = - cos(heading_diff) * gs_int
                # y_dif = gs_own - sin(heading_diff) * gs_int
                
                # denormalize
                x_dif = self.safe_vals["x_difference_speed"] * AC_SPD
                y_dif = self.safe_vals["y_difference_speed"] * AC_SPD
                gs_own = bs.traf.gs[ac_idx]
                
                # tan(heading_diff) = sin(heading_diff) / cos(heading_diff)
                # sin(heading_diff) ~ (gs_own - y_dif)
                # cos(heading_diff) ~ -x_dif
                
                heading_diff_rad = np.arctan2(gs_own - y_dif, -x_dif)
                heading_diff_deg = np.rad2deg(heading_diff_rad)
                
                # heading_diff = hdg_own - hdg_int
                # hdg_int = hdg_own - heading_diff
                heading_global_deg = (bs.traf.hdg[ac_idx] - heading_diff_deg) % 360

                heading_end_x = ((np.sin(np.deg2rad(heading_global_deg)) * ac_length)/max_distance)*self.unwrapped.window_width
                heading_end_y = ((np.cos(np.deg2rad(heading_global_deg)) * ac_length)/max_distance)*self.unwrapped.window_width

                # draw centered line for the aircraft
                pygame.draw.line(canvas,
                    color,
                    (x_pos - heading_end_x/2, y_pos + heading_end_y/2),
                    (x_pos + heading_end_x/2, y_pos - heading_end_y/2),
                    width = 4
                )

                # draw heading line
                heading_length = 15
                heading_end_x = ((np.sin(np.deg2rad(heading_global_deg)) * heading_length)/max_distance)*self.unwrapped.window_width
                heading_end_y = ((np.cos(np.deg2rad(heading_global_deg)) * heading_length)/max_distance)*self.unwrapped.window_width

                pygame.draw.line(canvas,
                    color,
                    (x_pos,y_pos),
                    ((x_pos)+heading_end_x,(y_pos)-heading_end_y),
                    width = 1
                )
            
            # Draw circle at the calculated position (center of the aircraft)
            pygame.draw.circle(
                canvas, 
                color,
                (x_pos,y_pos),
                radius = (INTRUSION_DISTANCE*NM2KM/max_distance)*self.unwrapped.window_width,
                width = 2
            )

        # draw intruders
        ac_length = 3

        for i in range(NUM_INTRUDERS):
            int_idx = i+1
            int_hdg = bs.traf.hdg[int_idx]
            heading_end_x = ((np.sin(np.deg2rad(int_hdg)) * ac_length)/max_distance)*self.unwrapped.window_width
            heading_end_y = ((np.cos(np.deg2rad(int_hdg)) * ac_length)/max_distance)*self.unwrapped.window_width

            int_qdr, int_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], bs.traf.lat[int_idx], bs.traf.lon[int_idx])

            # # determine color
            # if int_dis < INTRUSION_DISTANCE:
            #     color = (220,20,60)
            # else: 
            #     color = (80,80,80)
            
            
            
            
            if shap_values is not None:
                
                if i < len(shap_values.values[0]):
                    color = self._get_saliency_color(shap_values.values[0][i],np.max(np.abs(shap_values.values)),shap_values.base_values[0][0])
                else:
                    color = (80,80,80)
            else:
                color = (80,80,80)
            

            x_pos = (self.unwrapped.window_width/2)+(np.sin(np.deg2rad(int_qdr))*(int_dis * NM2KM)/max_distance)*self.unwrapped.window_width
            y_pos = (self.unwrapped.window_height/2)-(np.cos(np.deg2rad(int_qdr))*(int_dis * NM2KM)/max_distance)*self.unwrapped.window_height

            pygame.draw.line(canvas,
                color,
                (x_pos,y_pos),
                ((x_pos)+heading_end_x,(y_pos)-heading_end_y),
                width = 4
            )

            # draw heading line
            heading_length = 10
            heading_end_x = ((np.sin(np.deg2rad(int_hdg)) * heading_length)/max_distance)*self.unwrapped.window_width
            heading_end_y = ((np.cos(np.deg2rad(int_hdg)) * heading_length)/max_distance)*self.unwrapped.window_width

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
                radius = (INTRUSION_DISTANCE*NM2KM/max_distance)*self.unwrapped.window_width,
                width = 2
            )
            
            # draw a small number indicating intruder index
            index_text = self.font.render(str(i+1), True, (0, 0, 0))
            canvas.blit(index_text, (x_pos + 5, y_pos + 5))
        
        


        # draw target waypoint
        for qdr, dis, reach in zip(self.unwrapped.wpt_qdr, self.unwrapped.waypoint_distance, self.unwrapped.wpt_reach):

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
            

        self._post_render(canvas)