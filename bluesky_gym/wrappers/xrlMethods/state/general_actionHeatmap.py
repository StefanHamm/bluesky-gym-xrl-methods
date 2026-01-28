import numpy as np
import pygame
import copy
import bluesky as bs
from bluesky_gym.wrappers.xrlMethods.state.xrl_base_class import xrlBaseWrapper
from bluesky_gym.utils.constants import NM2KM,D_HEADING


class ActionHeatmapV1Wrapper(xrlBaseWrapper):
    
    def __init__(self, env,debug=False, export_gifs_path=None, fps=5,model=None):
        """
        Initialize the SaliencyHorizontalControl wrapper.

        Args:
            env: The Gym environment to wrap.
            debug (bool, optional): If True, enables debug mode for additional visualization and logging.
            export_gifs_path (str, optional): Directory path to export GIFs of episodes. If None, GIFs are not saved.
            fps (int, optional): Frames per second for GIF export and rendering. Default is 5.
            model: The trained model used for action predictions.

        """
        super().__init__(env,export_gifs_path,fps)
        

        self.DEBUG = debug
    
        # create working directory for gif creation
        
        self.episode_counter = 0
        self.step_counter = 0
        self.model = model

        self.font = pygame.font.SysFont(None, 24)
        self.frame_saved = False # Dont want to save all intermediate frames when exporting gifs
        self.d_heading = None

    def _build_observation_at_offset(self,offset_x_nm, offset_y_nm,waypoint_pos=None):
        """
        Calculates observation by temporarily moving the aircraft to the offset position.
        offset_x_nm: Right offset in NM (relative to aircraft heading)
        offset_y_nm: Forward offset in NM (relative to aircraft heading)
        waypoint_pos: Optional waypoint position to set heading towards (lat, lon) else it keeps current heading
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
        if waypoint_pos is not None:
            wpt_lat, wpt_lon = waypoint_pos
        
            # Calculate bearing from new position to waypoint
            new_hdg, _ = bs.tools.geo.kwikqdrdist(new_lat, new_lon, wpt_lat, wpt_lon)
        
        else:
            new_hdg = orig_hdg
        
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
        
        returnDict = {
            "observation": obs,
            "lat": new_lat,
            "lon": new_lon,
            "hdg": new_hdg
            
        }

        return returnDict
    
    def _create_action_grid(self,waypoint_pos=None):
        """
        Creates a grid of observations around the current aircraft position.
        Returns a 2D list of observations corresponding to grid positions.
        waypoint_pos: Optional waypoint position to set heading towards (lat, lon) else it keeps current heading
        """
        half_grid = self.grid_size // 2
        observations_grid = []
        
        for y in range(-half_grid, half_grid + 1):
            row = []
            for x in range(-half_grid, half_grid + 1):
                offset_x_nm = x * self.grid_spacing_km / NM2KM  # Convert km to NM
                offset_y_nm = y * self.grid_spacing_km / NM2KM  # Convert km to NM
                pos_obs = self._build_observation_at_offset(offset_x_nm, offset_y_nm,waypoint_pos)
                row.append(pos_obs)
            observations_grid.append(row)
        
        return observations_grid
    
    def _compute_action_heatmap(self,waypoint_pos=None):
        """
        Computes the action heatmap over the defined grid.
        Returns a 2D numpy array of action values.
        waypoint_pos: Optional waypoint position to set heading towards (lat, lon) else it keeps current heading
        """
        observations_grid = self.create_action_grid(waypoint_pos)
        # Flatten observations for batch prediction
        flat_obs = [item["observation"] for row in observations_grid for item in row]
        
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
                
                new_hdg = obs[3] + act_val * self.d_heading
                obs["new_hdg"] = new_hdg
                obs["action"] = act_val
                
        
        return observations_grid
    
    def _draw_action_heatmap(self,canvas,heatmap,observation_grid)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                pos = observation_grid[i][j]
                lat, lon, hdg = heatmap[i, j]
                
                x_pos, y_pos = self.lat_lon_to_screen_coordinates(lat, lon)
                new_hdg = hdg