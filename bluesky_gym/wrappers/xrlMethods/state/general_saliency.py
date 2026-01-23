import gymnasium as gym
import numpy as np
import pygame
import os
import copy
import imageio
import bluesky as bs

from bluesky_gym.utils.constants import NM2KM, MPS2KT
from pygame import font

class GeneralSaliency(gym.Wrapper):
    
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
            self.safe_obs = None
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
        self.num_intruders = 0
        self.action_frequency = 0
        self.distance_margin = 0.0
        
        
    def _create_safe_observation(self,obs):
        for key in self.safe_vals.keys():
            if key in obs:
                obs[key] = np.array([self.safe_vals[key]] *  self.num_intruders)
        return obs
    
    def _update_safe_observation(self,safe_obs,obs):
        for key in obs.keys():
            if key not in self.safe_vals:
                safe_obs[key] = obs[key]
        return safe_obs
        
        
        
    def _action_rollout_path(self):
        # copy the simulator
        ac_idx = bs.traf.id2idx('KL001')
        for step in range(30):  # simulate 100 steps ahead
            obs = self.unwrapped._get_obs()
            action = self.model.predict(obs, deterministic=True)[0]
            self.unwrapped._get_action(action)
            for i in range(self.action_frequency):
                bs.sim.step()
                # store ownship state in path_coordinates
               
            self.path_coordinates.append((bs.traf.lat[ac_idx], bs.traf.lon[ac_idx]))
            # if last coordinate is close to waypoint, stop
            index = 0
            for distance in self.unwrapped.waypoint_distance:
                if distance < self.distance_margin and self.unwrapped.wpt_reach[index] != 1:
                    return
    
    def _action_rollout_safe_state_path(self):
        ac_idx = bs.traf.id2idx('KL001')
        obs = self.unwrapped._get_obs()
        safe_obs = self._create_safe_observation(safe_obs)
        
        
        for step in range(30):  # simulate 100 steps ahead
            obs = self.unwrapped._get_obs()
            safe_obs = self._update_safe_observation(safe_obs,obs)
            action = self.model.predict(safe_obs, deterministic=True)[0]
            self.unwrapped._get_action(action)
            for i in range(self.action_frequency): #double the steps
                
                bs.sim.step()
            self.safe_action_path.append((bs.traf.lat[ac_idx], bs.traf.lon[ac_idx]))
            for distance in self.unwrapped.waypoint_distance:
                if distance < self.action_frequency:
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
        
    def _pre_render(self):
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
                
    def _post_render(self,canvas):
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
                
    def _get_saliency_color(self,shap_value,max_abs_shap_value, baseline_value):
        color = (0,0,0)
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
    
    def _draw_intruder_speed_bar(self,canvas,shap_value,x_pos,y_pos,bar_width=4,bar_height=20):
        speed_t = max(-2, min(2, shap_value))/2  # scale to -1 to +1
        
        bar_color = (255, 0, 0) if speed_t > 0 else (0, 0, 255)
        
        pygame.draw.line(canvas,
            bar_color,
            (x_pos + 10, y_pos),
            (x_pos + 10, y_pos - speed_t * bar_height),
            width = bar_width
        )
        
        # draw a rectangle around the speed bar 
        bar_rec_x = x_pos + 10 - bar_height//2
        bar_rec_y = y_pos - bar_height
        pygame.draw.rect(canvas,
            (0,0,0),
            (bar_rec_x, bar_rec_y, bar_width+1, bar_height * 2),
            width = 1
        )
        
    def _seperation_distance(self,ac1_idx, ac2_idx):
        lat1, lon1 = bs.traf.lat[ac1_idx], bs.traf.lon[ac1_idx]
        lat2, lon2 = bs.traf.lat[ac2_idx], bs.traf.lon[ac2_idx]
        
        separation = bs.tools.geo.kwikdist(lat1, lon1, lat2, lon2)
        return separation
        
    def _calculate_xpos_ypos(self,lat,lon,*args,**kwargs)->tuple:
        # This method should should convert the lat/lon to x/y positions on the pygame canvas
        # Since it depends on the specific environment and rendering setup, we leave it unimplemented here.
        # The user should implement this method in the subclass.
        
        raise NotImplementedError("This method needs to be implemented in the subclass.")
        
    def _draw_path(self,canvas,color,path_coordinates,ac_idx):
         for i,coord in enumerate(path_coordinates):
                if i == 0:
                    continue
                prev_coord = path_coordinates[i-1]
                lat1, lon1 = prev_coord
                lat2, lon2 = coord
                
                x_pos1,y_pos1 = self._calculate_xpos_ypos(lat1,lon1)

                
                x_pos2,y_pos2 = self._calculate_xpos_ypos(lat2,lon2)
                #print(x_pos1,y_pos1,x_pos2,y_pos2)
                pygame.draw.line(canvas,
                    color,
                    (x_pos1,y_pos1),
                    (x_pos2,y_pos2),
                    width = 2
                )
    
    def _draw_waypoints(self,canvas,waypoints_lat:list,waypoints_lon:list,waypoints_reach:list,color=(0,255,0)):
        for lat,lon,reach in zip(waypoints_lat,waypoints_lon,waypoints_reach):
            if reach:
                color = (155,155,155)
            else:
                color = (255,255,255)

            self._draw_single_waypoint_with_color(canvas,lat,lon,color)
            
    def _draw_single_waypoint_with_color(self,canvas,lat,lon,color=(0,255,0)):
        x_pos,y_pos = self._calculate_xpos_ypos(lat,lon)

        pygame.draw.circle(
            canvas, 
            color,
            (x_pos,y_pos),
            radius = 4,
            width = 0
        )
        
        pygame.draw.circle(
            canvas, 
            color,
            (x_pos,y_pos),
            radius = 7, # check if this radius is ok
            width = 2
        )
        
    def _draw_intruders_default(self,canvas,number_intruders):
        for int_idx in range(1,number_intruders+1):
            lat,long = bs.traf.lat[int_idx], bs.traf.lon[int_idx]
           
            
    def _draw_single_intruder_with_color(self,canvas,,color=(255,0,0)):
        
        
        
    def _draw_action_bar(self,canvas,shap_sum,x_pos,y_pos,bar_length,bar_height,orientation="horizontal",pos_text="+",neg_text="-",title_text="PLACEHOLDER"):
        
       
        for i in range(bar_length):
            # Scale from -1 (left) to +1 (right)
            value = (i / bar_length) * 2 - 1
            val = max(-1, min(1, value))
            if val < 0:
                t = -val
                color = (int(80 * (1-t)), int(80 * (1-t)), int(80 * (1-t) + 255 * t))
            else:
                t = val
                color = (int(80 * (1-t) + 255 * t), int(80 * (1-t)), int(80 * (1-t)))
                
            if orientation == "horizontal":
                pygame.draw.line(canvas, color, (x_pos + i, y_pos), (x_pos + i, y_pos + bar_height-3), 1)
            elif orientation == "vertical":
                pygame.draw.line(canvas, color, (x_pos, y_pos + i), (x_pos + bar_height-3, y_pos + i), 1)
        
        pos_text = font.render(pos_text, True, (0,0,0))
        neg_text = font.render(neg_text, True, (0,0,0))
        title_text = font.render(title_text, True, (0,0,0))
        
        if orientation == "horizontal":
            
            shap_sum += 2
            shap_sum = (shap_sum / 4) * bar_length  # scale to bar length
            pygame.draw.line(canvas, (0,0,0), (x_pos + int(shap_sum), y_pos), (x_pos + int(shap_sum), y_pos + bar_height-3), 3)
            pygame.draw.rect(canvas, (0,0,0), (x_pos, y_pos, bar_length, bar_height), 2)

            #draw text
            
            canvas.blit(neg_text, (x_pos - 10, y_pos + bar_height + 5))
            canvas.blit(pos_text, (x_pos + bar_length - 50, y_pos + bar_length + 5))
            canvas.blit(title_text, (x_pos, y_pos - 20))

        elif orientation == "vertical":
            shap_sum += 2
            shap_sum = (shap_sum / 4) * bar_height  # scale to bar length
            pygame.draw.line(canvas, (0,0,0), (x_pos, y_pos + int(shap_sum)), (x_pos + bar_height-3, y_pos + int(shap_sum)), 3)
            pygame.draw.rect(canvas, (0,0,0), (x_pos, y_pos, bar_height, bar_length), 2)
            
            #draw text
            canvas.blit(neg_text, (x_pos + bar_height + 5, y_pos + bar_length - 10))
            canvas.blit(pos_text, (x_pos + bar_height + 5, y_pos))
            canvas.blit(title_text, (x_pos, y_pos - 20))
            
    