import gymnasium as gym
import numpy as np
import pygame
from bluesky_gym.envs.sector_cr_env import AC_DENSITY_MU, AC_DENSITY_SIGMA, AC_DENSITY_RANGE, NUM_AC_STATE, ACTION_FREQUENCY, INTRUSION_DISTANCE, NM2KM, D_HEADING, AC_SPD,ACTOR,CENTER
from bluesky_gym.envs.common.screen_dummy import ScreenDummy
from bluesky_gym.wrappers.xrlMethods.state.general_saliency import SaliencyMapV1Wrapper
import bluesky_gym.envs.common.functions as fn
import os
import imageio
import bluesky as bs


# This wrapper creates saliency maps from the current observation
#class SaliencyMapV1Wrapper(gym.ObservationWrapper):





class SaliencySectorControl(SaliencyMapV1Wrapper):
    
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
        super().__init__(env, safe_vals, debug, export_gifs_path, fps, color_mode, plot_action_path, plot_safe_path, model)
        self.action_frequency = ACTION_FREQUENCY #needs to be set since its used inside the general_saliency wrapper but is a global variable in all envs
        self.distance_margin = INTRUSION_DISTANCE # same for this one
        self.num_intruders = NUM_AC_STATE
        self.d_hdg = D_HEADING
        self.px_per_km = 200
        self.ownshiplatlon = []
    
    def _calculate_xpos_ypos(self,lat,lon,*args,**kwargs)->tuple:
        qdr, dis = bs.tools.geo.kwikqdrdist(CENTER[0],CENTER[1], lat, lon)
        
        x_pos = (self.unwrapped.window_width/2)+(np.sin(np.deg2rad(qdr))*(dis * NM2KM)*self.px_per_km)
        y_pos = (self.unwrapped.window_height/2)-(np.cos(np.deg2rad(qdr))*(dis * NM2KM)*self.px_per_km)
        
        return x_pos,y_pos
        
            
            
    def reset(self, seed=None, options=None):
        obs,inf = super().reset(seed=seed)
        self.episode_counter += 1
        self.step_counter = 0
        self.ownshiplatlon = []
        
        if self.export_gifs_path is not None:
            # create folder inside frames for this episode
            self.episode_frames_path = os.path.join(self.frames_path, f"episode_{self.episode_counter}")
            os.makedirs(self.episode_frames_path, exist_ok=True)
    
        
        if self.plot_action_path and self.model is not None:
            self._calculate_projected_path(safe=False,has_waypoints=False)
            

        if self.render_mode == "human":
            self._render_frame()

        return obs,inf
            
    def step(self, action, shap_values=None,examplePlane = None):
        if self.plot_safe_path and self.model is not None and self.safe_vals is not None:
             self._calculate_projected_path(safe=True,has_waypoints=False)
        
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
        reward= self.unwrapped._get_reward()
        truncate = self.unwrapped._check_inside_airspace()
        info = self.unwrapped._get_info()

        if truncate:
            self.export_episode_gif()

        return observation, reward, False, truncate, info
    
    
    
    def _render_frame(self,shap_values=None,examplePlane=None):
        self._pre_render()
        max_distance = max(np.linalg.norm(point1 - point2) for point1 in self.unwrapped.poly_points for point2 in self.unwrapped.poly_points)*NM2KM
        self.px_per_km = self.unwrapped.window_width/max_distance
        
        canvas = pygame.Surface(self.unwrapped.window_size)
        canvas.fill((135,206,235))
        
        if self.plot_action_path and self.model is not None:
            self._draw_path(canvas,(255,0,0),self.path_coordinates)
           
                
        if self.plot_safe_path and self.model is not None and self.safe_vals is not None:
            self._draw_path(canvas,(255,0,255),self.safe_action_path)
                
        
        # Draw airspace
        airspace_color = (255, 0, 0)
        coords = [((self.unwrapped.window_width/2)+point[1]*NM2KM*self.px_per_km, (self.unwrapped.window_height/2)-point[0]*NM2KM*self.px_per_km) for point in self.unwrapped.poly_points]
        pygame.draw.polygon(canvas, airspace_color, coords, width=2)

        # draw ownship
        ac_idx = bs.traf.id2idx(ACTOR)
        ac_length = 10
        ac_hdg = bs.traf.hdg[ac_idx]
        heading_end_x = np.sin(np.deg2rad(ac_hdg)) * ac_length
        heading_end_y = np.cos(np.deg2rad(ac_hdg)) * ac_length
        ac_qdr, ac_dis = bs.tools.geo.kwikqdrdist(CENTER[0], CENTER[1], bs.traf.lat[ac_idx], bs.traf.lon[ac_idx])

        x_pos = (self.unwrapped.window_width/2)+(np.sin(np.deg2rad(ac_qdr))*(ac_dis * NM2KM)*self.px_per_km)
        y_pos = (self.unwrapped.window_height/2)-(np.cos(np.deg2rad(ac_qdr))*(ac_dis * NM2KM)*self.px_per_km)
        
        pygame.draw.line(canvas,
            (0,0,0),
            (x_pos,y_pos),
            ((x_pos)+heading_end_x,(y_pos)-heading_end_y),
            width = 4
        )

        #Draw heading line
        heading_length = 20
        heading_end_x = np.sin(np.deg2rad(ac_hdg)) * heading_length
        heading_end_y = np.cos(np.deg2rad(ac_hdg)) * heading_length

        pygame.draw.line(canvas,
                (0,0,0),
                (x_pos,y_pos),
                ((x_pos)+heading_end_x,(y_pos)-heading_end_y),
                width = 1
        )

        # Plot additionally the intended_heading (baseline heading)
        if shap_values is not None and self.DEBUG:

                
            base_val  = shap_values.base_values[0][0]

            # intended_heading = current_heading + baseline_turn
            intended_heading = bs.traf.hdg[ac_idx] + base_val * D_HEADING
            
            heading_end_y = np.cos(np.deg2rad(intended_heading)) * ac_length
            heading_end_x = np.sin(np.deg2rad(intended_heading)) * ac_length
            # Draw baseline/intended heading as a Green line
            pygame.draw.line(canvas,
                (0,255,0),
                (x_pos,y_pos),
                ((x_pos)+heading_end_x,(y_pos)-heading_end_y),
                width = 2
            )

        if self.DEBUG:
            # Ghost Intruder Parameters
            # Use safe_vals as the ghost data
            if examplePlane is not None:
                ghost_dict = examplePlane
            else:
                ghost_dict = self.safe_vals
            
            ghost_color = (255, 0, 255)  # Magenta for ghost intruder

            # Denormalize relative position (x_r=North, y_r=East in SectorCREnv)
            d_north_m = ghost_dict["x_r"] * 13000
            d_east_m = ghost_dict["y_r"] * 13000
            
            # Helper: px_per_m
            px_per_m = self.px_per_km / 1000.0

            # Calculate Ghost Screen Position relative to Ownship
            # Ownship is at (x_pos, y_pos)
            # ghost_x (East) = x_pos + d_east_m
            # ghost_y (North) = y_pos - d_north_m (Screen Y is inverted)
            ghost_x = x_pos + d_east_m * px_per_m
            ghost_y = y_pos - d_north_m * px_per_m

            # Draw Ghost Circle (Intruder)
            pygame.draw.circle(
                canvas,
                ghost_color,
                (ghost_x, ghost_y),
                radius=INTRUSION_DISTANCE*NM2KM*self.px_per_km, 
                width=4
            )

            # Calculate Ghost Heading
            # Reconstruct velocity vectors
            ac_tas = bs.traf.tas[ac_idx]
            # Ownship velocity (North, East)
            v_north_own = np.cos(np.deg2rad(ac_hdg)) * ac_tas
            v_east_own = np.sin(np.deg2rad(ac_hdg)) * ac_tas
            
            # Relative velocity from ghost_dict
            v_north_rel = ghost_dict["vx_r"] * 32
            v_east_rel = ghost_dict["vy_r"] * 66
            
            # Ghost absolute velocity
            v_north_ghost = v_north_own + v_north_rel
            v_east_ghost = v_east_own + v_east_rel
            
            # Ghost Heading
            # atan2(y, x) -> atan2(East, North) for compass heading
            ghost_hdg_rad = np.arctan2(v_east_ghost, v_north_ghost)
            ghost_hdg_deg = np.degrees(ghost_hdg_rad)
            
            # Draw Heading Line
            heading_length = 20
            # Heading 0 is Up (-Y in pygame)
            heading_end_x_ghost = np.sin(np.deg2rad(ghost_hdg_deg)) * heading_length
            heading_end_y_ghost = np.cos(np.deg2rad(ghost_hdg_deg)) * heading_length
            
            pygame.draw.line(canvas,
                ghost_color,
                (ghost_x, ghost_y),
                (ghost_x + heading_end_x_ghost, ghost_y - heading_end_y_ghost),
                width=6
            )
            
            
            # --- Second Heading Calculation (using sin/cos track) ---
            # 1. Reconstruct relative velocity vector from sin(track) and cos(track)
            # The magnitude of relative velocity is needed. Using the one derived from vx_r/vy_r
            speed_rel = np.sqrt(v_north_rel**2 + v_east_rel**2)
            
            sin_track = ghost_dict["sin(track)"]
            cos_track = ghost_dict["cos(track)"]
            
            # track angle was defined as atan2(vy_rel, vx_rel) -> atan2(East, North)
            # which implies vx_rel ~ cos(track), vy_rel ~ sin(track)
            v_north_rel_2 = cos_track * speed_rel
            v_east_rel_2 = sin_track * speed_rel
            
            # 2. Add ownship velocity to get absolute velocity
            v_north_ghost_2 = v_north_own + v_north_rel_2
            v_east_ghost_2 = v_east_own + v_east_rel_2
            
            # 3. Calculate heading
            ghost_hdg_rad_2 = np.arctan2(v_east_ghost_2, v_north_ghost_2)
            ghost_hdg_deg_2 = np.degrees(ghost_hdg_rad_2)
            
            heading_end_x_ghost_2 = np.sin(np.deg2rad(ghost_hdg_deg_2)) * heading_length
            heading_end_y_ghost_2 = np.cos(np.deg2rad(ghost_hdg_deg_2)) * heading_length
            
            # Plot second heading in Cyan
            pygame.draw.line(canvas,
                (0, 255, 255), # Cyan
                (ghost_x, ghost_y),
                (ghost_x + heading_end_x_ghost_2, ghost_y - heading_end_y_ghost_2),
                width=4
            )
            
            # check if the distance is caluclated correctly 
            calculated_distance = np.sqrt(d_north_m**2 + d_east_m**2)
            ghost_distance = ghost_dict["distances"] * 15000 + 50000
            distance_error = abs(calculated_distance - ghost_distance)
            print(f"DEBUG Ghost Distance Error (m): {distance_error:.2f}")
            
            
        # draw intruders
        ac_length = 3
        
        ac_loc = fn.latlong_to_nm(CENTER, np.array([bs.traf.lat[ac_idx], bs.traf.lon[ac_idx]])) * NM2KM * 1000 
        distances = [fn.euclidean_distance(ac_loc, fn.latlong_to_nm(CENTER, np.array([bs.traf.lat[i], bs.traf.lon[i]])) * NM2KM * 1000) for i in range(1, self.unwrapped.num_ac)]
        
        # Sort indices by distance to know which ones are in the observation
        # Note: ac_idx_by_dist contains indices relative to the 'distances' list (0 to N-1), 
        # which corresponds to intruder IDs (1 to N).
        sorted_indices = np.argsort(distances)
        top_k_indices = sorted_indices[:NUM_AC_STATE] # Indices of the closest 4 intruders
        obs_rank_map = {idx + 1: rank for rank, idx in enumerate(top_k_indices)}

        for i in range(self.unwrapped.num_ac -1):
            int_idx = i+1
            int_hdg = bs.traf.hdg[int_idx]
            heading_end_x = np.sin(np.deg2rad(int_hdg)) * ac_length
            heading_end_y = np.cos(np.deg2rad(int_hdg)) * ac_length

            int_qdr, int_dis = bs.tools.geo.kwikqdrdist(CENTER[0], CENTER[1], bs.traf.lat[int_idx], bs.traf.lon[int_idx])
            separation = bs.tools.geo.kwikdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], bs.traf.lat[int_idx], bs.traf.lon[int_idx])
            
            
            x_pos = (self.unwrapped.window_width/2)+(np.sin(np.deg2rad(int_qdr))*(int_dis * NM2KM)*self.px_per_km)
            y_pos = (self.unwrapped.window_height/2)-(np.cos(np.deg2rad(int_qdr))*(int_dis * NM2KM)*self.px_per_km)
            color = (80,80,80)
            
            if shap_values is not None:
                if int_idx in obs_rank_map:
                    rank = obs_rank_map[int_idx]
                    if rank < len(shap_values.values[0]):
                        control_shap = shap_values.values[0][rank][0]
                        max_saliency = np.max(np.abs(shap_values.values[0][:,0]))
                        speed_shap = shap_values.values[0][rank][1]
                        control_baseline = shap_values.base_values[0][0]
                        color = self._get_saliency_color(control_shap,max_saliency,control_baseline)    
                        self._draw_intruder_speed_bar(canvas,speed_shap, x_pos, y_pos)

            
            pygame.draw.line(canvas,
                color,
                (x_pos,y_pos),
                ((x_pos)+heading_end_x,(y_pos)-heading_end_y),
                width = 4
            )

            # draw heading line
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
                radius = INTRUSION_DISTANCE*NM2KM*self.px_per_km,
                width = 2
            )

            # import code
            # code.interact(local=locals())
        
        
        # Draw legend for SHAP influence
        padding = 40
        legend_x = padding
        legend_y = self.unwrapped.window_size[1] - 80
        legend_width = 150

      
            
        if shap_values is not None:
            shap_sums = np.sum(shap_values.values[0], axis=0)

            if self.DEBUG:
                
                action_taken = self.last_action
                baseline_value = shap_values.base_values[0]
                self._draw_debug_menue(canvas,legend_x,legend_y-legend_width,action_taken,shap_sums,baseline_value)


            # Position for bottom-left (taking labels into account)

            cross_y = self.unwrapped.window_size[1] - legend_width - padding
            
            self._draw_shap_cross(canvas, legend_x, cross_y, legend_width, shap_sums, neg_labels=["Left","Dec"], pos_labels=["Right","Inc"],title="Turn & Speed Influences")

        self._post_render(canvas)