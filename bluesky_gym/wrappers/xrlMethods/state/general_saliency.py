import numpy as np
import pygame
import copy
import bluesky as bs
from .xrl_base_class import xrlBaseWrapper




class SaliencyMapV1Wrapper(xrlBaseWrapper):
    
    def __init__(self, env, safe_vals=None, debug=False, export_gifs_path=None, fps=5, color_mode="clipped",model=None):
        """
        Initialize the SaliencyHorizontalControl wrapper.

        Args:
            env: The Gym environment to wrap.
            safe_vals (dict, optional): Initial safe values for debugging and visualization.
            debug (bool, optional): If True, enables debug mode for additional visualization and logging.
            export_gifs_path (str, optional): Directory path to export GIFs of episodes. If None, GIFs are not saved.
            fps (int, optional): Frames per second for GIF export and rendering. Default is 5.
            color_mode (str, optional): Color mode for saliency visualization. Use "quantitized", "clipped", or "scaled".

        """
        super().__init__(env,export_gifs_path,fps)
        
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
            self.safe_obs = None
        # create working directory for gif creation
        
        self.episode_counter = 0
        self.step_counter = 0
        self.model = model
        self.path_coordinates = []
        self.safe_action_path = []
        self.font = pygame.font.SysFont(None, 24)
        self.frame_saved = False # Dont want to save all intermediate frames when exporting gifs
        
        
    def _create_safe_observation(self,obs):
        safe_obs = copy.deepcopy(obs)
        for key in self.safe_vals.keys():
            if key in safe_obs and key in obs:
                safe_obs[key] = np.array([self.safe_vals[key]] *  self.num_intruders)
            else:
                print(f"Key {key} not found in observation.")
        return safe_obs
    
    def _update_safe_observation(self,safe_obs,obs):
        for key in obs.keys():
            if key not in self.safe_vals:
                safe_obs[key] = obs[key]
        return safe_obs
        
    def _calculate_projected_path(self,safe=False,has_waypoints=False):
        
        prev_state = self._save_traffic_state()
        if safe:
            self.safe_action_path = []
        else:
            self.path_coordinates = []
        self._simulate_rollout(safe,has_waypoints)
        self._restore_traffic_state(prev_state)
        self.unwrapped._get_obs() #reset internal obs state

    def _simulate_rollout(self,safe=False,has_waypoints=False):
        # copy the simulator
        ac_idx = bs.traf.id2idx('KL001')
        obs = self.unwrapped._get_obs()
        if safe:
            safe_obs = self._create_safe_observation(obs)
        max_steps = 30
        if safe:
            max_steps = 20
        else:
            max_steps = 300
            
        
        for step in range(max_steps):  # simulate 100 steps ahead
            obs = self.unwrapped._get_obs()
            if safe:
                obs = self._update_safe_observation(safe_obs,obs)
            action = self.model.predict(obs, deterministic=True)[0]
            self.unwrapped._get_action(action)
            for i in range(self.action_frequency):
                bs.sim.step()
                # store ownship state in path_coordinates
            
            if safe:
                self.safe_action_path.append((bs.traf.lat[ac_idx], bs.traf.lon[ac_idx]))
            else:
                self.path_coordinates.append((bs.traf.lat[ac_idx], bs.traf.lon[ac_idx]))
            # if last coordinate is close to waypoint, stop
            index = 0
            if has_waypoints:
                for distance in self.unwrapped.waypoint_distance:
                    if distance < self.distance_margin and self.unwrapped.wpt_reach[index] != 1:
                        return
            reward, terminated = self.unwrapped._get_reward()
            if terminated:
                return
        
    def _get_saliency_color(self,shap_value,max_abs_shap_value, baseline_value) -> tuple[int,int,int]:
        color = (80,80,80)
        if self.color_mode == self.color_map["quantitized"]:
            val = np.clip(shap_value, -1, 1)
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
                val = np.clip(shap_value, -1, 1)
            elif self.color_mode == self.color_map["scaled"]:
                val = shap_value
                #scale by max abs value of shap values
                val = val / max_abs_shap_value if max_abs_shap_value != 0 else 0
            elif self.color_mode == self.color_map["baseline_scaled"]:
                val = shap_value
                scale_factor = 1+ abs(baseline_value)
                val = val/scale_factor
            elif self.color_mode == self.color_map["default"]:
                val = shap_value / 2.0
            
            if val < 0:
                t = -val
                color = (int(80 * (1-t)), int(80 * (1-t)), int(80 * (1-t) + 255 * t))
            else:
                t = val
                color = (int(80 * (1-t) + 255 * t), int(80 * (1-t)), int(80 * (1-t)))
        return color
    
    def _draw_debug_menue(self,canvas,x_pos,y_pos,action_taken:list,shap_sums:list=None,baseline_values:list=None):
        formatted_sums = ", ".join(f"{x:.3f}" for x in shap_sums)
        formatted_baselines = ", ".join(f"{x:.3f}" for x in baseline_values)
        formatted_action_taken = ", ".join(f"{x:.3f}" for x in action_taken)
        sum_text = self.font.render(f"Sum of SHAP values: {formatted_sums}", True, (0,0,0))
        canvas.blit(sum_text, (x_pos, y_pos - 50))
        baseline_text = self.font.render(f"Baseline: {formatted_baselines}", True, (0,0,0))
        canvas.blit(baseline_text, (x_pos, y_pos - 70))
        action_taken_text = self.font.render(f"Action taken: {formatted_action_taken}", True, (0,0,0))
        canvas.blit(action_taken_text, (x_pos, y_pos - 90))
        legend_text = self.font.render("Green line: Heading w/o other aircrafts", True, (0,100,0))
        canvas.blit(legend_text, (x_pos, y_pos - 110))
    
    def _draw_intruder_speed_bar(self,canvas,shap_value,x_pos,y_pos,thickness=4,length=20):
        speed_t = max(-2, min(2, shap_value))/2  # scale to -1 to +1
        
        bar_color = (255, 0, 0) if speed_t > 0 else (0, 0, 255)
        
        pygame.draw.line(canvas,
            bar_color,
            (x_pos + 10, y_pos),
            (x_pos + 10, y_pos - speed_t * length),
            width = thickness
        )
        
        # draw a rectangle around the speed bar 
        bar_rec_x = x_pos + 10 - thickness//2
        bar_rec_y = y_pos - length
        pygame.draw.rect(canvas,
            (0,0,0),
            (bar_rec_x, bar_rec_y, thickness+1, length * 2),
            width = 1
        )
        
    def _seperation_distance(self,ac1_idx, ac2_idx):
        lat1, lon1 = bs.traf.lat[ac1_idx], bs.traf.lon[ac1_idx]
        lat2, lon2 = bs.traf.lat[ac2_idx], bs.traf.lon[ac2_idx]
        
        separation = bs.tools.geo.kwikdist(lat1, lon1, lat2, lon2)
        return separation
        
    def _draw_path(self,canvas,color,path_coordinates,skip_first=False):
         for i,coord in enumerate(path_coordinates):
                if i == 0:
                    if skip_first:
                        continue
                    #use current agent position as previous coord
                    agent_idx = bs.traf.id2idx('KL001')
                    prev_coord = bs.traf.lat[agent_idx], bs.traf.lon[agent_idx]
                else:
                    prev_coord = path_coordinates[i-1]
                lat1, lon1 = prev_coord
                lat2, lon2 = coord
                
                x_pos1,y_pos1 = self.lat_lon_to_screen_coordinates (lat1,lon1)

                
                x_pos2,y_pos2 = self.lat_lon_to_screen_coordinates (lat2,lon2)
                #print(x_pos1,y_pos1,x_pos2,y_pos2)
                pygame.draw.line(canvas,
                    color,
                    (x_pos1,y_pos1),
                    (x_pos2,y_pos2),
                    width = 2
                )
   
    def _draw_shap_bar(self,canvas,shap_sum,x_pos,y_pos,bar_length,thickness,orientation="horizontal",pos_text="+",neg_text="-",title_text="PLACEHOLDER"):
        
       
        self._draw_gradient_bar(canvas, (x_pos,y_pos), bar_length, thickness,orientation == "horizontal")
       
        pos_text = self.font.render(pos_text, True, (0,0,0))
        neg_text = self.font.render(neg_text, True, (0,0,0))
        title_text = self.font.render(title_text, True, (0,0,0))
        
        # Get dimensions
        pos_w, pos_h = pos_text.get_size()
        neg_w, neg_h = neg_text.get_size()
        title_w, title_h = title_text.get_size()
        
        if orientation == "horizontal":
            
            shap_sum += 2
            shap_sum = (shap_sum / 4) * bar_length  # scale to bar length
            pygame.draw.line(canvas, (0,0,0), (x_pos + int(shap_sum), y_pos), (x_pos + int(shap_sum), y_pos + thickness-3), 3)
            pygame.draw.rect(canvas, (0,0,0), (x_pos, y_pos, bar_length, thickness), 2)

            #draw text
            
            canvas.blit(neg_text, (x_pos , y_pos + thickness + 5))
            canvas.blit(pos_text, (x_pos + bar_length - pos_w, y_pos + thickness + 5))
            canvas.blit(title_text, (x_pos + (bar_length- title_w)//2, y_pos - 20))

        elif orientation == "vertical":
            shap_sum += 2
            shap_sum = (shap_sum / 4) * thickness  # scale to bar length
            pygame.draw.line(canvas, (0,0,0), (x_pos, y_pos + int(shap_sum)), (x_pos + thickness-3, y_pos + int(shap_sum)), 3)
            pygame.draw.rect(canvas, (0,0,0), (x_pos, y_pos, thickness, bar_length), 2)
            
            #draw text
            canvas.blit(neg_text, (x_pos + thickness + 5, y_pos + bar_length - 10))
            canvas.blit(pos_text, (x_pos + thickness + 5, y_pos))
            canvas.blit(title_text, (x_pos, y_pos - 20))
            
    def _draw_shap_circle(self, canvas, x_pos, y_pos, radius, shap_sums, neg_labels:list=["L","-"], pos_labels:list=["R","+"]):
        """
        Draws a circular vector plot representing combined Heading and Speed influence.
        shaps_sums[0] -> Heading (X-axis)
        shaps_sums[1] -> Speed   (Y-axis)
        """
        # Center of the circle
        cx = x_pos + radius
        cy = y_pos + radius

        # Draw background and border
        pygame.draw.circle(canvas, (240, 240, 240), (cx, cy), radius) # Light grey filled
        pygame.draw.circle(canvas, (0, 0, 0), (cx, cy), radius, 2)    # Black border

        # Draw Axes (Crosshairs)
        pygame.draw.line(canvas, (160, 160, 160), (cx - radius, cy), (cx + radius, cy), 1)
        pygame.draw.line(canvas, (160, 160, 160), (cx, cy - radius), (cx, cy + radius), 1)

        # Draw Grid Rings (optional, e.g. at 50% intensity)
        pygame.draw.circle(canvas, (200, 200, 200), (cx, cy), int(radius * 0.5), 1)

        # --- Calculate Vector Position ---
        # Assuming shap sums are roughly in range [-2, 2] like in the bar plots
        range_val = 2.0
        
        # Heading (X-Axis): Negative = Left, Positive = Right
        h_val = shap_sums[0]
        
        # Speed (Y-Axis): Negative = Slow Down, Positive = Speed Up
        # In Pygame, Y increases downwards, so we invert Y for "Up" to mean "Speed Up"
        s_val = shap_sums[1]

        # Map to pixels relative to center
        rel_x = (h_val / range_val) * radius
        rel_y = -(s_val / range_val) * radius # Note the minus for Y inversion

        # Clamp vector to be inside the circle
        magnitude = np.sqrt(rel_x**2 + rel_y**2)
        if magnitude > radius:
            scale = radius / magnitude
            rel_x *= scale
            rel_y *= scale

        px = cx + rel_x
        py = cy + rel_y

        # Draw Vector Line
        pygame.draw.line(canvas, (0, 0, 0), (cx, cy), (px, py), 3)

        # Draw Vector Head (Dot)
        # You can color this dot based on magnitude or keep it simple red
        pygame.draw.circle(canvas, (255, 0, 0), (int(px), int(py)), 6)

        # --- Labels ---
        offset = 15
        
        # Function to render centered text
        def draw_label(text, x, y):
            surf = self.font.render(text, True, (0, 0, 0))
            w, h = surf.get_size()
            canvas.blit(surf, (x - w/2, y - h/2))

        # X-Axis Labels (Heading)
        draw_label(neg_labels[0], cx - radius - offset, cy) # Left
        draw_label(pos_labels[0], cx + radius + offset, cy) # Right
        
        # Y-Axis Labels (Speed)
        draw_label(pos_labels[1], cx, cy - radius - offset) # Up (Increase)
        draw_label(neg_labels[1], cx, cy + radius + offset) # Down (Decrease)

    def _draw_gradient_bar(self, canvas, start_pos, length, thickness, horizontal=True):
        """
        Draws a gradient bar (Blue -> Grey -> Red) on the canvas.
        """
        for i in range(length):
            # Normalize i to [-1, 1] for color calculation
            value = (i / length) * 2 - 1
            val = max(-1, min(1, value))
            
            # Get color
            if val < 0:
                t = -val
                color = (int(80 * (1-t)), int(80 * (1-t)), int(80 * (1-t) + 255 * t))
            else:
                t = val
                color = (int(80 * (1-t) + 255 * t), int(80 * (1-t)), int(80 * (1-t)))

            # Draw slice
            if horizontal:
                # x varies, y is constant block
                pygame.draw.line(canvas, color, (start_pos[0] + i, start_pos[1]), 
                                              (start_pos[0] + i, start_pos[1] + thickness-3), 1)
            else:
                # y varies, x is constant block
                pygame.draw.line(canvas, color, (start_pos[0], start_pos[1] + length - i), 
                                              (start_pos[0] + thickness-3, start_pos[1] + length - i), 1)

    def _draw_shap_cross(self, canvas, x_pos, y_pos, length, shap_sums, thickness=20, neg_labels=["L","-"], pos_labels=["R","+"],title="Default title"):
        """
        Draws two overlapping SHAP bars in a cross shape with gradients.
        shap_sums[0] -> Horizontal Bar (Heading)
        shap_sums[1] -> Vertical Bar (Speed)
        """
        cx = x_pos + length // 2
        cy = y_pos + length // 2
        half_len = length // 2
        
        # --- Draw Bars ---
        # 1. Horizontal Bar (Heading)
        h_bar_x = cx - half_len
        h_bar_y = cy - thickness // 2
        self._draw_gradient_bar(canvas, (h_bar_x, h_bar_y), length, thickness, horizontal=True)

        # 2. Vertical Bar (Speed) - Drawn directly over center
        v_bar_x = cx - thickness // 2
        v_bar_y = cy - half_len
        # To avoid overdrawing the intersection weirdly, we can just draw it. 
        # Alternatively, blending could be used, but standard drawing is usually fine for "crosshairs".
        self._draw_gradient_bar(canvas, (v_bar_x, v_bar_y), length, thickness, horizontal=False)
        
        # --- Draw Outlines ---
        pygame.draw.rect(canvas, (0,0,0), (h_bar_x, h_bar_y, length, thickness), 2)
        pygame.draw.rect(canvas, (0,0,0), (v_bar_x, v_bar_y, thickness, length), 2)

        # --- Draw Indicators (Black lines for current value) ---
        range_val = 2.0
        
        # Horizontal Indicator
        h_val = np.clip(shap_sums[0], -range_val, range_val)
        # Map [-2, 2] to [0, length]
        h_px = ((h_val + range_val) / (2 * range_val)) * length
        pygame.draw.line(canvas, (0,0,0), (h_bar_x + int(h_px), h_bar_y ), 
                                          (h_bar_x + int(h_px), h_bar_y + thickness), 3)

        # Vertical Indicator
        # Note: Input data usually maps larger value -> top.
        # But here y increases downwards. 
        # So -2 (bottom) -> y = length, +2 (top) -> y = 0
        s_val = np.clip(shap_sums[1], -range_val, range_val)
        # Invert s_val for display so positive is "Up" (lower Y pixel value)
        # s_val = -2 => pixel = length (bottom)
        # s_val = +2 => pixel = 0 (top)
        # Formula: (1 - (val + 2)/4) * length  => (2-val)/4 * length
        v_px = ((range_val - s_val) / (2 * range_val)) * length
        pygame.draw.line(canvas, (0,0,0), (v_bar_x , v_bar_y + int(v_px)), 
                                          (v_bar_x + thickness , v_bar_y + int(v_px)), 3)

        # --- Labels ---
        offset = 25
        font_surf = self.font.render("A", True, (0,0,0)) # Dummy render to get height
        
        def draw_label(text, x, y):
            surf = self.font.render(text, True, (0, 0, 0))
            w, h = surf.get_size()
            canvas.blit(surf, (x - w/2, y - h/2))

        # Horizontal Labels
        draw_label(neg_labels[0], h_bar_x - offset, cy)
        draw_label(pos_labels[0], h_bar_x + length + offset, cy)

        # Vertical Labels
        draw_label(pos_labels[1], cx, v_bar_y - offset)
        draw_label(neg_labels[1], cx, v_bar_y + length + offset)
        
        # Title
        title_surf = self.font.render(title, True, (0,0,0))
        title_w, title_h = title_surf.get_size()
        canvas.blit(title_surf, (cx - title_w/2, y_pos - title_h - 1.5*offset))
    