import gymnasium as gym
import numpy as np
import pygame
from bluesky_gym.envs.horizontal_cr_env import D_HEADING,ACTION_FREQUENCY,NUM_INTRUDERS,NM2KM,INTRUSION_DISTANCE,DISTANCE_MARGIN,AC_SPD,WAYPOINT_DISTANCE_MAX
import bluesky as bs
from bluesky_gym.envs.common.screen_dummy import ScreenDummy
import bluesky_gym.envs.common.functions as fn
import os
import copy
import imageio


# This wrapper creates saliency maps from the current observation
#class SaliencyMapV1Wrapper(gym.ObservationWrapper):





class SaliencyHorizontalControl(gym.Wrapper):
    
    def __init__(self, env, safe_vals=None, debug=False, export_gifs_path=None, fps=5, color_mode="clipped",plot_action_path=False,plot_safe_path=False,model=None):
        """
        Initialize the SaliencyHorizontalControl wrapper.

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
        self.plot_action_path = plot_action_path
        self.plot_safe_path = plot_safe_path
        self.model = model
        self.path_coordinates = []
        self.safe_action_path = []
        
        
    def _action_rollout_path(self):
        # copy the simulator
        ac_idx = bs.traf.id2idx('KL001')
        for step in range(100):  # simulate 100 steps ahead
            obs = self.unwrapped._get_obs()
            action = self.model.predict(obs, deterministic=True)[0]
            self.unwrapped._get_action(action)
            for i in range(ACTION_FREQUENCY):
                bs.sim.step()
                # store ownship state in path_coordinates
               
            self.path_coordinates.append((bs.traf.lat[ac_idx], bs.traf.lon[ac_idx]))
            # if last coordinate is close to waypoint, stop
            index = 0
            for distance in self.unwrapped.waypoint_distance:
                if distance < DISTANCE_MARGIN and self.unwrapped.wpt_reach[index] != 1:
                    return
    
    def _action_rollout_safe_state_path(self):
        ac_idx = bs.traf.id2idx('KL001')
        safe_obs = self.unwrapped._get_obs()
        safe_obs = {
            "intruder_distance": np.array([self.safe_vals["dist"]] * NUM_INTRUDERS),
            "cos_difference_pos": np.array([self.safe_vals["cos"]] * NUM_INTRUDERS),
            "sin_difference_pos": np.array([self.safe_vals["sin"]] * NUM_INTRUDERS),
            "x_difference_speed": np.array([self.safe_vals["dx"]] * NUM_INTRUDERS),
            "y_difference_speed": np.array([self.safe_vals["dy"]] * NUM_INTRUDERS)
        }
        
        
        for step in range(100):  # simulate 100 steps ahead
            obs = self.unwrapped._get_obs()
            safe_obs["cos_drift"] = obs["cos_drift"]
            safe_obs["sin_drift"] = obs["sin_drift"]
            safe_obs["waypoint_distance"] = obs["waypoint_distance"]
            action = self.model.predict(safe_obs, deterministic=True)[0]
            self.unwrapped._get_action(action)
            for i in range(ACTION_FREQUENCY): #double the steps
                
                bs.sim.step()
            self.safe_action_path.append((bs.traf.lat[ac_idx], bs.traf.lon[ac_idx]))
            for distance in self.unwrapped.waypoint_distance:
                if distance < DISTANCE_MARGIN:
                    return
        
    
    def _save_traffic_state(self):
        return {
            "lat": np.copy(bs.traf.lat),
            "lon": np.copy(bs.traf.lon),
            "hdg": np.copy(bs.traf.hdg),
            "alt": np.copy(bs.traf.alt),
            "tas": np.copy(bs.traf.tas),
            "gs": np.copy(bs.traf.gs),
            "trk": np.copy(bs.traf.trk),
            "vs": np.copy(bs.traf.vs),
            "sim_time": bs.sim.simt
        }

    def _restore_traffic_state(self, state):
        bs.traf.lat[:] = state["lat"]
        bs.traf.lon[:] = state["lon"]
        bs.traf.hdg[:] = state["hdg"]
        bs.traf.alt[:] = state["alt"]
        bs.traf.tas[:] = state["tas"]
        bs.traf.gs[:] = state["gs"]
        bs.traf.trk[:] = state["trk"]
        bs.traf.vs[:] = state["vs"]
        bs.sim.simt = state["sim_time"]
            
            
    def reset(self, seed=None, options=None):
        obs,inf = super().reset(seed=seed)
        self.episode_counter += 1
        self.step_counter = 0
        
        if self.export_gifs_path is not None:
            # create folder inside frames for this episode
            self.episode_frames_path = os.path.join(self.frames_path, f"episode_{self.episode_counter}")
            os.makedirs(self.episode_frames_path, exist_ok=True)
    
        
        if self.plot_action_path and self.model is not None:
            self.path_coordinates = []
            # copy previous simulator state
            prev_state = self._save_traffic_state()
            self._action_rollout_path()
            self._restore_traffic_state(prev_state)
            # update obs after rollout
            self.unwrapped._get_obs()
            

        if self.unwrapped.render_mode == "human":
            self._render_frame()

        return obs,inf
            
    def step(self, action, shap_values=None,examplePlane = None):
        self.step_counter += 1
        if self.plot_safe_path and self.model is not None and self.safe_vals is not None:
            self.safe_action_path = []
            # copy previous simulator state
            prev_state = self._save_traffic_state()
            self._action_rollout_safe_state_path()
            self._restore_traffic_state(prev_state)
            self.unwrapped._get_obs() #resets observation to current state after rollout
        
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
                        "dist": observation["intruder_distance"][0],
                        "cos": observation["cos_difference_pos"][0],
                        "sin": observation["sin_difference_pos"][0],
                        "dx": observation["x_difference_speed"][0],
                        "dy": observation["y_difference_speed"][0]
                    }

                self._render_frame(shap_values=shap_values,examplePlane=examplePlane)

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
    
    def _draw_path(self,canvas,color,path_coordinates,ac_idx):
         for i,coord in enumerate(path_coordinates):
                if i == 0:
                    continue
                prev_coord = path_coordinates[i-1]
                lat1, lon1 = prev_coord
                lat2, lon2 = coord
                qdr1, dis1 = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], lat1, lon1)
                qdr2, dis2 = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], lat2, lon2)

                
                x_pos1 = (self.unwrapped.window_width/2)+(np.sin(np.deg2rad(qdr1))*(dis1 * NM2KM)/200)*self.unwrapped.window_width
                y_pos1 = (self.unwrapped.window_height/2)-(np.cos(np.deg2rad(qdr1))*(dis1 * NM2KM)/200)*self.unwrapped.window_height
                
                x_pos2 = (self.unwrapped.window_width/2)+(np.sin(np.deg2rad(qdr2))*(dis2 * NM2KM)/200)*self.unwrapped.window_width
                y_pos2 = (self.unwrapped.window_height/2)-(np.cos(np.deg2rad(qdr2))*(dis2 * NM2KM)/200)*self.unwrapped.window_height
                
                #print(x_pos1,y_pos1,x_pos2,y_pos2)
                pygame.draw.line(canvas,
                    color,
                    (x_pos1,y_pos1),
                    (x_pos2,y_pos2),
                    width = 2
                )
        
    
    def _render_frame(self,shap_values=None,examplePlane=None):
        if self.unwrapped.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.unwrapped.window = pygame.display.set_mode(self.unwrapped.window_size)

        if self.unwrapped.clock is None and self.render_mode == "human":
            self.unwrapped.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                if self.window is not None:
                    pygame.display.quit()
                self.close()
                
        ac_idx = bs.traf.id2idx('KL001')        
                  
        max_distance = 200 # width of screen in km

        canvas = pygame.Surface(self.unwrapped.window_size)
        canvas.fill((135,206,235))

        if self.plot_action_path and self.model is not None:
            self._draw_path(canvas,(255,0,0),self.path_coordinates,ac_idx)
           
                
        if self.plot_safe_path and self.model is not None and self.safe_vals is not None:
            self._draw_path(canvas,(255,0,255),self.safe_action_path,ac_idx)
                
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
        if shap_values is not None:

                
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
            if examplePlane is not None:
                self.safe_vals = examplePlane
            #plot one intrude (now plotted relative to ownship heading)
            color = (0,255,0)

            # compute relative bearing from cos/sin (these encode ac_hdg - qdr)
            rel_bearing_rad = np.arctan2(self.safe_vals["sin"], self.safe_vals["cos"])  # rel = ac_hdg - qdr
            rel_bearing_deg = np.rad2deg(rel_bearing_rad)
            # convert to global bearing from ownship to intruder
            int_qdr = (bs.traf.hdg[ac_idx] - rel_bearing_deg) % 360

            # CORRECT DISTANCE CALCULATION:
            # safe_vals["dist"] is normalized (0-1), so we multiply by MAX to get KM
            dist_km = self.safe_vals["dist"] * WAYPOINT_DISTANCE_MAX
            
            # Use consistent scale for X and Y to prevent distortion
            screen_scale = self.unwrapped.window_height 

            x_pos = (self.unwrapped.window_width/2) + (np.sin(np.deg2rad(int_qdr)) * dist_km / max_distance) * screen_scale
            y_pos = (self.unwrapped.window_height/2) - (np.cos(np.deg2rad(int_qdr)) * dist_km / max_distance) * screen_scale

            # compute intruder heading: rotate local (dx,dy) by ownship heading
            heading_mag = np.sqrt(self.safe_vals["dx"]**2 + self.safe_vals["dy"]**2)
            if heading_mag > 1e-8:
                # REVERSE TRANSFORM SPEED DIFFERENCE TO HEADING
                # x_dif = - cos(heading_diff) * gs_int
                # y_dif = gs_own - sin(heading_diff) * gs_int
                
                # denormalize
                x_dif = self.safe_vals["dx"] * AC_SPD
                y_dif = self.safe_vals["dy"] * AC_SPD
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
                    saliency = shap_values.values[0][i]
                    
                    
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

            # import code
            # code.interact(local=locals())

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
        font = pygame.font.SysFont(None, 24)

        # Draw sum of SHAP values above the legend
        if shap_values is not None:
            try:
                shap_sum = float(np.sum(shap_values.values))
            except Exception:
                shap_sum = float(np.sum(shap_values))
            sum_text = font.render(f"Sum of SHAP values: {shap_sum:.3f}", True, (0,0,0))
            canvas.blit(sum_text, (legend_x, legend_y - 30))
            baseline_text = font.render(f"Baseline: {shap_values.base_values[0][0]:.3f}", True, (0,0,0))
            canvas.blit(baseline_text, (legend_x, legend_y - 50))
            action_taken_text = font.render(f"Action taken: {self.last_action}", True, (0,0,0))
            text_rect = action_taken_text.get_rect()
            x = int(self.unwrapped.window_width / 2 - text_rect.width / 2)
            y = int(self.unwrapped.window_height / 2 - 30 - text_rect.height)
            canvas.blit(action_taken_text, (legend_x, legend_y - 70))
            #canvas.blit(action_taken_text, (x, y))
            
            legend_text = font.render("Green line: Heading w/o other aircrafts", True, (0,100,0))
            canvas.blit(legend_text, (legend_x, legend_y - 90))

            intended_heading = self.unwrapped.ac_hdg + shap_values.base_values[0][0] * D_HEADING
            

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
            pygame.draw.line(canvas, color, (legend_x + i, legend_y), (legend_x + i, legend_y + legend_height-3), 1)
        
        #draw sum of shap bar
        if shap_values is not None:
            shap_sum = float(np.sum(shap_values.values))
            shap_sum += 2 # to scale from -2 to + 2 to 0 to 4
            shap_sum = (shap_sum / 4) * legend_width
            pygame.draw.line(canvas, (0,0,0), (legend_x + int(shap_sum), legend_y), (legend_x + int(shap_sum), legend_y + legend_height-3), 3)
            

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