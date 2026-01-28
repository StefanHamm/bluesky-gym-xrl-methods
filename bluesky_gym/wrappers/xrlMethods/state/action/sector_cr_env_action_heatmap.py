import gymnasium as gym
import numpy as np
import pygame
from bluesky_gym.envs.sector_cr_env import AC_DENSITY_MU, AC_DENSITY_SIGMA, AC_DENSITY_RANGE, NUM_AC_STATE, ACTION_FREQUENCY, INTRUSION_DISTANCE, NM2KM, D_HEADING, AC_SPD,ACTOR,CENTER
import bluesky as bs
from bluesky_gym.envs.common.screen_dummy import ScreenDummy
import bluesky_gym.envs.common.functions as fn
import os
import imageio

import time


# This wrapper creates saliency maps from the current observation
#class SaliencyMapV1Wrapper(gym.ObservationWrapper):





class ActionHeatmapWrapper(gym.Wrapper):
    
    def __init__(self, env, model,draw_action_heatmap=True, grid_size=5, grid_spacing_km=10, export_gifs_path=None, fps=5, **kwargs):
        super().__init__(env, **kwargs)
        self.heatmap_model = model
        self.grid_size = grid_size        # e.g., 5x5 grid
        self.grid_spacing_km = grid_spacing_km 
        self.draw_action_heatmap = draw_action_heatmap
        self.fps = fps

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
        self.episode_counter += 1
        self.step_counter = 0
        
        if self.export_gifs_path is not None:
            # create folder inside frames for this episode
            self.episode_frames_path = os.path.join(self.frames_path, f"episode_{self.episode_counter}")
            os.makedirs(self.episode_frames_path, exist_ok=True)
        
        return super().reset(seed=seed, options=options)

    def build_observation_at_offset(self,offset_x_nm, offset_y_nm):
        """
        Calculates observation by temporarily moving the aircraft to the offset position.
        offset_x_nm: Right offset in NM (relative to aircraft heading)
        offset_y_nm: Forward offset in NM (relative to aircraft heading)
        """
        
        # 1. Save Original State
        ac_idx = bs.traf.id2idx('KL001')
        orig_lat = bs.traf.lat[ac_idx]
        orig_lon = bs.traf.lon[ac_idx]
        orig_hdg = bs.traf.hdg[ac_idx]
        
        # 2. Calculate New Coordinates
        # Convert local offset (Right/Forward) to global (North/East) using current heading
        hdg = bs.traf.hdg[ac_idx]
        rad_hdg = np.deg2rad(hdg)
        cos_h = np.cos(rad_hdg)
        sin_h = np.sin(rad_hdg)
        
        # North = Forward * cos(h) - Right * sin(h)
        # East  = Forward * sin(h) + Right * cos(h)
        d_north_nm = offset_y_nm * cos_h - offset_x_nm * sin_h
        d_east_nm  = offset_y_nm * sin_h + offset_x_nm * cos_h
        
        # Apply offset (1 deg lat = 60 nm)
        # This approximation is valid for small local offsets
        new_lat = orig_lat + (d_north_nm / 60.0)
        # 1 deg lon = 60 nm * cos(lat)
        new_lon = orig_lon + (d_east_nm / (60.0 * np.cos(np.deg2rad(orig_lat))))

        # Calculate new heading towards waypoint
        # Assuming single waypoint or taking the first one
        #wpt_lat = self.unwrapped.wpt_lat[0]
        #wpt_lon = self.unwrapped.wpt_lon[0]
        
        # Calculate bearing from new position to waypoint
        #new_hdg, _ = bs.tools.geo.kwikqdrdist(new_lat, new_lon, wpt_lat, wpt_lon)
        new_hdg = orig_hdg  # Keep heading same for simplicity
        
        
        # 3. Teleport & Get Observation
        try:
            bs.traf.lat[ac_idx] = new_lat
            bs.traf.lon[ac_idx] = new_lon
            bs.traf.hdg[ac_idx] = new_hdg
            
            # Use the environment's internal method to get observation
            # This ensures perfect consistency with training
            obs = self.unwrapped._get_obs()
        finally:
            # 4. Restore Original State (CRITICAL)
            bs.traf.lat[ac_idx] = orig_lat
            bs.traf.lon[ac_idx] = orig_lon
            bs.traf.hdg[ac_idx] = orig_hdg
            self.unwrapped._get_obs()  # Refresh internal state
        

        return obs,new_lat,new_lon,new_hdg


    def create_action_grid(self):
        """
        Creates a grid of observations around the current aircraft position.
        Returns a 2D list of observations corresponding to grid positions.
        """
        half_grid = self.grid_size // 2
        observations_grid = []
        
        for y in range(-half_grid, half_grid + 1):
            row = []
            for x in range(-half_grid, half_grid + 1):
                offset_x_nm = x * self.grid_spacing_km / NM2KM  # Convert km to NM
                offset_y_nm = y * self.grid_spacing_km / NM2KM  # Convert km to NM
                pos_obs = self.build_observation_at_offset(offset_x_nm, offset_y_nm)
                row.append(pos_obs)
            observations_grid.append(row)
        
        return observations_grid
    
    def step(self, action):
        
        self.step_counter += 1

        self.unwrapped._get_action(action)

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
    
    def compute_action_heatmap(self):
        """
        Computes the action heatmap over the defined grid.
        Returns a 2D numpy array of action values.
        """
        observations_grid = self.create_action_grid()
        heatmap = np.zeros((self.grid_size, self.grid_size), dtype=object)
        
        # Flatten observations for batch prediction
        flat_obs = [item[0] for row in observations_grid for item in row]
        
        # Vectorize Dict observation: List of Dicts -> Dict of Arrays
        batch_obs = {}
        if len(flat_obs) > 0:
            for key in flat_obs[0].keys():
                batch_obs[key] = np.stack([obs[key] for obs in flat_obs])
        
        # Batch predict
        actions, _ = self.heatmap_model.predict(batch_obs, deterministic=True)
        
        k = 0
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                obs = observations_grid[i][j]
                # action from batch
                act_val = actions[k][0] if isinstance(actions[k], (list, np.ndarray)) else actions[k]
                k += 1
                
                new_hdg = obs[3] + act_val * D_HEADING
                heatmap[i, j] = (obs[1],obs[2],new_hdg)  # Assuming single action output
        
        return heatmap
    
    
    def _render_frame(self):
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
        
        ac_idx = bs.traf.id2idx('KL001')
        max_distance = max(np.linalg.norm(point1 - point2) for point1 in self.unwrapped.poly_points for point2 in self.unwrapped.poly_points)*NM2KM

        px_per_km = self.unwrapped.window_width/max_distance
        canvas = pygame.Surface(self.unwrapped.window_size)
        canvas.fill((135,206,235))
        
        # Draw airspace
        airspace_color = (255, 0, 0)
        coords = [((self.unwrapped.window_width/2)+point[1]*NM2KM*px_per_km, (self.unwrapped.window_height/2)-point[0]*NM2KM*px_per_km) for point in self.unwrapped.poly_points]
        pygame.draw.polygon(canvas, airspace_color, coords, width=2)

        if self.draw_action_heatmap:
            # heading change is action times D_HEADING
            observations_grid = self.create_action_grid()
            
            # Prepare batch
            flat_obs = [item[0] for row in observations_grid for item in row]
            
            # Vectorize Dict observation: List of Dicts -> Dict of Arrays
            batch_obs = {}
            if len(flat_obs) > 0:
                for key in flat_obs[0].keys():
                    batch_obs[key] = np.stack([obs[key] for obs in flat_obs])
            
            # Batch predict
            actions, _ = self.heatmap_model.predict(batch_obs, deterministic=True)
            
            k = 0
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    obs = observations_grid[i][j]
                    
                    act_val = actions[k][0] if isinstance(actions[k], (list, np.ndarray)) else actions[k]
                    k += 1
                    
                    new_hdg = obs[3] + act_val * D_HEADING
                    lat, lon = obs[1], obs[2]
                    # draw a small arrow at position (i,j) with heading new_hdg
                    int_qdr, int_dis = bs.tools.geo.kwikqdrdist(CENTER[0],CENTER[1], lat, lon)

                    x_pos = (self.unwrapped.window_width/2)+(np.sin(np.deg2rad(int_qdr))*(int_dis * NM2KM)/max_distance)*self.unwrapped.window_width
                    y_pos = (self.unwrapped.window_height/2)-(np.cos(np.deg2rad(int_qdr))*(int_dis * NM2KM)/max_distance)*self.unwrapped.window_height
                    arrow_length = 15
                    # Arrow direction
                    dy = (np.cos(np.deg2rad(new_hdg)) * arrow_length)
                    dx = (np.sin(np.deg2rad(new_hdg)) * arrow_length)
                    # Arrow color: shade of red for positive, shade of blue for negative
                    intensity = int(255 * np.clip(abs(act_val), 0, 1))
                    color = (intensity, 0, 0) if act_val > 0 else (0, 0, intensity)
                    # Main arrow line
                    pygame.draw.line(canvas,
                        color,
                        (x_pos, y_pos),
                        (x_pos + dx, y_pos - dy),
                        width=2
                    )
                    # Arrowhead (V shape)
                    head_len = 6
                    head_angle = 25 # degrees
                    left_hdg = new_hdg + 180 - head_angle
                    right_hdg = new_hdg + 180 + head_angle
                    left_dy = np.cos(np.deg2rad(left_hdg)) * head_len
                    left_dx = np.sin(np.deg2rad(left_hdg)) * head_len
                    right_dy = np.cos(np.deg2rad(right_hdg)) * head_len
                    right_dx = np.sin(np.deg2rad(right_hdg)) * head_len
                    # Left side
                    pygame.draw.line(canvas,
                        color,
                        (x_pos + dx, y_pos - dy),
                        (x_pos + dx + left_dx, y_pos - dy - left_dy),
                        width=2
                    )
                    # Right side
                    pygame.draw.line(canvas,
                        color,
                        (x_pos + dx, y_pos - dy),
                        (x_pos + dx + right_dx, y_pos - dy - right_dy),
                        width=2
                    )

        

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