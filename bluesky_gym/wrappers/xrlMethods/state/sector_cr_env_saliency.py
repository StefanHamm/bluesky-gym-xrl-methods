import gymnasium as gym
import numpy as np
import pygame
from bluesky_gym.envs.sector_cr_env import AC_DENSITY_MU, AC_DENSITY_SIGMA, AC_DENSITY_RANGE, NUM_AC_STATE, ACTION_FREQUENCY, INTRUSION_DISTANCE, NM2KM, D_HEADING, AC_SPD,ACTOR,CENTER
from bluesky_gym.envs.common.screen_dummy import ScreenDummy
import bluesky_gym.envs.common.functions as fn
import os
import imageio
import bluesky as bs


# This wrapper creates saliency maps from the current observation
#class SaliencyMapV1Wrapper(gym.ObservationWrapper):





class SaliencySectorControl(gym.Wrapper):
    
    def __init__(self, env, safe_vals=None, debug=False, export_gifs_path=None, fps=5, color_mode="clipped"):
        """
        Initialize the SaliencySectorControl wrapper.

        Args:
            env: The Gym environment to wrap.
            safe_vals (dict, optional): Initial safe values for debugging and visualization.
            debug (bool, optional): If True, enables debug mode for additional visualization and logging.
            export_gifs_path (str, optional): Directory path to export GIFs of episodes. If None, GIFs are not saved.
            fps (int, optional): Frames per second for GIF export and rendering. Default is 5.
            color_mode (str, optional): Color mode for saliency visualization. Use "quantitized", "clipped", or "scaled".

        Sets up rendering, debugging, and GIF export directories. Initializes episode and step counters.
        """
        super().__init__(env)
        self.fps = fps
        #self.unwrapped.window_size=(1024,1024)
        self.last_action = None  
        self.DEBUG = debug
        self.color_map = {
            "quantitized": 1,
            "clipped": 2,
            "scaled": 3,
            "default": 4,
            "baseline_scaled": 5
        }
        self.color_mode = self.color_map.get(color_mode, 2)  # Default to "clipped" if invalid mode provided
        if safe_vals is not None:
            self.safe_vals = safe_vals
        # create working directory for gif creation
        self.export_gifs_path = export_gifs_path
        if self.export_gifs_path is not None:
            os.makedirs(self.export_gifs_path, exist_ok=True)
        # inside create two folder: frames and gifs
        if self.export_gifs_path is not None:
            self.frames_path = os.path.join(self.export_gifs_path, "frames")
            self.gifs_path = os.path.join(self.export_gifs_path, "gifs")
            os.makedirs(self.frames_path, exist_ok=True)
            os.makedirs(self.gifs_path, exist_ok=True)
        self.episode_counter = 0
        self.step_counter = 0
        
        
        
            
            
    def reset(self, seed=None, options=None):
        # 1. Let the environment reset itself (generates polygon, aircraft, etc.)
        observation, info = super().reset(seed=seed,options=options)
        
        # 2. Update wrapper-specific counters and paths
        self.episode_counter += 1
        self.step_counter = 0
        
        if self.export_gifs_path is not None:
            self.episode_frames_path = os.path.join(self.frames_path, f"episode_{self.episode_counter}")
            os.makedirs(self.episode_frames_path, exist_ok=True)
            
        return observation, info
            
    def step(self, action, shap_values=None,examplePlane = None):
        self.step_counter += 1
        
        self.unwrapped._get_action(action)
        self.last_action = action  # Store the last action

        action_frequency = ACTION_FREQUENCY
        for i in range(action_frequency):
            bs.sim.step()
            if self.render_mode == "human":
                observation = self.unwrapped._get_obs()

                # In debug mode, we update the examplePlane with the latest observation
                # to ensure the ghost intruder stays synchronized with the simulation step.
                if self.DEBUG and examplePlane is not None:
                    examplePlane = {
                        "x_r": observation["x_r"][0],
                        "y_r": observation["y_r"][0],
                        "vx_r": observation["vx_r"][0],
                        "vy_r": observation["vy_r"][0],
                        "cos(track)": observation["cos(track)"][0],
                        "sin(track)": observation["sin(track)"][0],
                        "distances": observation["distances"][0]
                    }

                self._render_frame(shap_values=shap_values,examplePlane=examplePlane)

        observation = self.unwrapped._get_obs()        
        reward = self.unwrapped._get_reward()
        info = self.unwrapped._get_info()

        # truncate instead of terminate to avoid aircraft learning to exit sector fast
        truncate = self.unwrapped._check_inside_airspace()

        # bluesky reset?? bs.sim.reset()
        if truncate:
            if self.export_gifs_path is not None:
                # export gif from saved frames
                gif_filename = os.path.join(self.gifs_path, f"episode_{self.episode_counter}.gif")
                images = [imageio.imread(os.path.join(self.episode_frames_path, f"frame_{step}.png")) for step in range(1, self.step_counter + 1)]
                imageio.mimsave(gif_filename, images, fps=self.fps)

        return observation, reward, False, truncate, info
    
    def export_gif(self):
        if self.export_gifs_path is not None:
                # export gif from saved frames
                gif_filename = os.path.join(self.gifs_path, f"episode_{self.episode_counter}.gif")
                images = [imageio.imread(os.path.join(self.episode_frames_path, f"frame_{step}.png")) for step in range(1, self.step_counter + 1)]
                imageio.mimsave(gif_filename, images, fps=self.fps)
    
    def _render_frame(self,shap_values=None,examplePlane=None):
        if self.unwrapped.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.unwrapped.window = pygame.display.set_mode(self.unwrapped.window_size)

        if self.unwrapped.clock is None and self.render_mode == "human":
            self.unwrapped.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                if self.unwrapped.window is not None:
                    pygame.display.quit()
                self.close()
                
        max_distance = max(np.linalg.norm(point1 - point2) for point1 in self.unwrapped.poly_points for point2 in self.unwrapped.poly_points)*NM2KM
        px_per_km = self.unwrapped.window_width/max_distance
        
        canvas = pygame.Surface(self.unwrapped.window_size)
        canvas.fill((135,206,235))
        
        # Draw airspace
        airspace_color = (255, 0, 0)
        coords = [((self.unwrapped.window_width/2)+point[1]*NM2KM*px_per_km, (self.unwrapped.window_height/2)-point[0]*NM2KM*px_per_km) for point in self.unwrapped.poly_points]
        pygame.draw.polygon(canvas, airspace_color, coords, width=2)

        # draw ownship
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
        if shap_values is not None:

                
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
            px_per_m = px_per_km / 1000.0

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
                radius=INTRUSION_DISTANCE*NM2KM*px_per_km, 
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
            
            
            
            
            if shap_values is not None:
                if int_idx in obs_rank_map:
                    rank = obs_rank_map[int_idx]
                    if rank < len(shap_values.values[0]):
                        saliency = shap_values.values[0][rank][0]
                        speed_inflence = shap_values.values[0][rank][1]
                        
                        
                        if self.color_mode == self.color_map["quantitized"]:
                            val = np.clip(saliency, -1, 1)
                            if val < -0.75:
                                color = (0, 0, 255)        # Dark blue (very strong left)
                            elif val < -0.5:
                                color = (50, 100, 255)     # Blue (strong left)
                            elif val < -0.25:
                                color = (100, 150, 255)    # Light blue (moderate left)
                            elif val < -0.125:
                                color = (150, 180, 255)    # Pale blue (weak left)
                            elif val < 0.125:
                                color = (80, 80, 80)       # Grey (minimal influence)
                            elif val < 0.25:
                                color = (255, 200, 100)    # Pale orange (weak right)
                            elif val < 0.5:
                                color = (255, 165, 0)      # Orange (moderate right)
                            elif val < 0.75:
                                color = (255, 100, 0)      # Dark orange (strong right)
                            else:
                                color = (255, 0, 0)        # Red (very strong right)
                                                
                        else:
                            if self.color_mode == self.color_map["clipped"]:
                                val = np.clip(saliency, -1, 1)
                            elif self.color_mode == self.color_map["scaled"]:
                                val = saliency
                                #scale by max abs value of shap values
                                max_abs = np.max(np.abs(shap_values.values))
                                val = val / max_abs if max_abs != 0 else 0
                            elif self.color_mode == self.color_map["baseline_scaled"]:
                                val = saliency
                                baseline = shap_values.base_values[0][0]
                                scale_factor = 1+ abs(baseline)
                                val = val/scale_factor
                            elif self.color_mode == self.color_map["default"]:
                                val = saliency / 2.0
                            
                            if val < 0:
                                t = -val
                                color = (int(80 * (1-t)), int(80 * (1-t)), int(80 * (1-t) + 255 * t))
                            else:
                                t = val
                                color = (int(80 * (1-t) + 255 * t), int(80 * (1-t)), int(80 * (1-t)))
                                
                            # create a a bar to the right of the intruder indicating speed influence.
                            # it ranges from -0.5 (blue, decrease speed) to +0.5 (red, increase speed)
                            # it should be a vertical bar of length 20 pixels and width 4 pixels
                            # if the speed is positive (increase speed), the bar goes up, else it goes down 
                            speed_bar_length = 20
                            speed_bar_width = 4
                            speed_t = max(-0.5, min(0.5, speed_inflence)) * 2  # scale to -1 to +1
                            
                            bar_color = (255, 0, 0) if speed_t > 0 else (0, 0, 255)
                            bar_x = (self.unwrapped.window_width/2)+(np.sin(np.deg2rad(int_qdr))*(int_dis * NM2KM)*px_per_km)
                            bar_y = (self.unwrapped.window_height/2)-(np.cos(np.deg2rad(int_qdr))*(int_dis * NM2KM)*px_per_km)
                            
                            pygame.draw.line(canvas,
                                bar_color,
                                (bar_x + 10, bar_y),
                                (bar_x + 10, bar_y - speed_t * speed_bar_length),
                                width = speed_bar_width
                            )
                            
                                
                    
                            
                else:
                    color = (80,80,80)
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
                radius = INTRUSION_DISTANCE*NM2KM*px_per_km,
                width = 2
            )

            # import code
            # code.interact(local=locals())
        
        
        # Draw legend for SHAP influence
        legend_x = 30
        legend_y = self.unwrapped.window_size[1] - 80
        legend_width = 200
        legend_height = 20
        font = pygame.font.SysFont(None, 24)

        # Draw sum of SHAP values above the legend
        if shap_values is not None:
            try:
                shap_sum = float(np.sum(shap_values.values[0][:,0]))
            except Exception:
                shap_sum = float(np.sum(shap_values))
            sum_text = font.render(f"Sum of SHAP values: {shap_sum:.3f}", True, (0,0,0))
            canvas.blit(sum_text, (legend_x, legend_y - 30))
            baseline_text = font.render(f"Baseline: {shap_values.base_values[0][0]:.3f}", True, (0,0,0))
            canvas.blit(baseline_text, (legend_x, legend_y - 50))
            
            action_taken_text = font.render(f"Action taken: {self.last_action[0]:.2f},{self.last_action[1]:.2f}", True, (0,0,0))
            text_rect = action_taken_text.get_rect()
            x = int(self.unwrapped.window_width / 2 - text_rect.width / 2)
            y = int(self.unwrapped.window_height / 2 - 30 - text_rect.height)
            canvas.blit(action_taken_text, (legend_x, legend_y - 70))
            #canvas.blit(action_taken_text, (x, y))
            
            legend_text = font.render("Green line: Heading w/o other aircrafts", True, (0,100,0))
            canvas.blit(legend_text, (legend_x, legend_y - 90))

            intended_heading = bs.traf.hdg[bs.traf.id2idx(ACTOR)] + shap_values.base_values[0][0] * D_HEADING
            

        # Draw color scale: left (blue) to right (red)
        for i in range(legend_width):
            # Scale from -1 (left) to +1 (right)
            value = (i / legend_width) * 2 - 1
            val = max(-1, min(1, value))
            if val < 0:
                t = -val
                color = (int(80 * (1-t)), int(80 * (1-t)), int(80 * (1-t) + 255 * t))
            else:
                t = val
                color = (int(80 * (1-t) + 255 * t), int(80 * (1-t)), int(80 * (1-t)))
            pygame.draw.line(canvas, color, (legend_x + i, legend_y), (legend_x + i, legend_y + legend_height), 1)

        # Draw border
        pygame.draw.rect(canvas, (0,0,0), (legend_x, legend_y, legend_width, legend_height), 2)

        # Add text labels
        left_text = font.render('Left', True, (0,0,0))
        right_text = font.render('Right', True, (0,0,0))
        canvas.blit(left_text, (legend_x - 10, legend_y + legend_height + 5))
        canvas.blit(right_text, (legend_x + legend_width - 50, legend_y + legend_height + 5))


        self.unwrapped.window.blit(canvas, canvas.get_rect())
        pygame.display.update()
        self.unwrapped.clock.tick(self.metadata["render_fps"])
        
        if self.export_gifs_path is not None:
            # save frame to episode frames folder use the current step count as filename
            frame_filename = os.path.join(self.episode_frames_path, f"frame_{self.step_counter}.png")
            try:
                pygame.image.save(canvas, frame_filename)
            except pygame.error as e:
                print(f"Error saving frame {self.step_counter} of episode {self.episode_counter}: {e}")