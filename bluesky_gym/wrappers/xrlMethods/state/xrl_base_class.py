import gymnasium as gym
import pygame
import os
import imageio
import bluesky as bs
import numpy as np






class xrlBaseWrapper(gym.Wrapper):
  
    def __init__(self, env,export_gifs_path=None,fps=5, xrl_rendering=True):
        """This is a base class for XRL wrappers and subclasses. This has common functinality like pre_render function, saving a frame, exporting a gif

        Args:
            env (gym.evn): env instance
            export_gifs_path (str, optional): String to export path, creates directories. Defaults to None.
            fps (int, optional): fps of the gif. Defaults to 5.
            xrl_rendering (bool, optional): Turn on or off the XRL specific rendering. Defaults to True.
        """ 
        super().__init__(env)
        self.export_gifs_path = export_gifs_path
        self._init_gif_folders()
        self.fps = fps
        self.path_coordinates = []
        self.safe_action_path = []
        self.xrl_rendering = xrl_rendering
        
    def _init_gif_folders(self):
        if self.export_gifs_path is not None:
            os.makedirs(self.export_gifs_path, exist_ok=True)
        # inside create two folder: frames and gifs
        if self.export_gifs_path is not None:
            self.frames_path = os.path.join(self.export_gifs_path, "frames")
            self.gifs_path = os.path.join(self.export_gifs_path, "gifs")
            os.makedirs(self.frames_path, exist_ok=True)
            os.makedirs(self.gifs_path, exist_ok=True)
            
    def export_episode_gif(self):
        if self.export_gifs_path is not None:
                # export gif from saved frames
                gif_filename = os.path.join(self.gifs_path, f"episode_{self.episode_counter}.gif")
                images = [imageio.imread(os.path.join(self.episode_frames_path, f"frame_{step}.png")) for step in range(1, self.step_counter + 1)]
                imageio.mimsave(gif_filename, images, fps=self.fps)

                
    def _post_render(self,canvas):
        self.unwrapped.window.blit(canvas, canvas.get_rect())
        pygame.display.update()
        self.unwrapped.clock.tick(self.metadata["render_fps"])

        if self.export_gifs_path is not None and not self.frame_saved:
            self.frame_saved = True
            # save frame to episode frames folder use the current step count as filename
            frame_filename = os.path.join(self.episode_frames_path, f"frame_{self.step_counter}.png")
            try:
                pygame.image.save(canvas, frame_filename)
            except pygame.error as e:
                print(f"Error saving frame {self.step_counter} of episode {self.episode_counter}: {e}")
    
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
                exit()
                
    def _save_traffic_state(self):
        return {
            # --- Basic Physics ---
            "lat": np.copy(bs.traf.lat),
            "lon": np.copy(bs.traf.lon),
            "hdg": np.copy(bs.traf.hdg),
            "alt": np.copy(bs.traf.alt),
            "tas": np.copy(bs.traf.tas),
            "cas": np.copy(bs.traf.cas),
            "gs": np.copy(bs.traf.gs),
            "trk": np.copy(bs.traf.trk),
            "vs": np.copy(bs.traf.vs),
            "sim_time": bs.sim.simt,

            # --- Kinematics (Hidden State) ---
            "ax": np.copy(bs.traf.ax),           # Current acceleration
            # CHANGE HERE: Use ap.turnphi instead of bank
            "turnphi": np.copy(bs.traf.ap.turnphi), # Current bank angle

            # --- Intermediate Guidance (The 'Switch' variables) ---
            "aporasas_tas": np.copy(bs.traf.aporasas.tas),
            "aporasas_alt": np.copy(bs.traf.aporasas.alt),
            "aporasas_vs":  np.copy(bs.traf.aporasas.vs),
            "aporasas_hdg": np.copy(bs.traf.aporasas.hdg),

            # --- Autopilot Intent ---
            "selspd": np.copy(bs.traf.selspd),
            "swlnav": np.copy(bs.traf.swlnav),
            "swvnav": np.copy(bs.traf.swvnav)
        }

    def _restore_traffic_state(self, state):
        # --- Restore Basic Physics ---
        bs.traf.lat[:] = state["lat"]
        bs.traf.lon[:] = state["lon"]
        bs.traf.hdg[:] = state["hdg"]
        bs.traf.alt[:] = state["alt"]
        bs.traf.tas[:] = state["tas"]
        bs.traf.cas[:] = state["cas"]
        bs.traf.gs[:]  = state["gs"]
        bs.traf.trk[:] = state["trk"]
        bs.traf.vs[:]  = state["vs"]
        bs.sim.simt    = state["sim_time"]

        # --- Restore Kinematics ---
        bs.traf.ax[:] = state["ax"]
        # CHANGE HERE: Restore to ap.turnphi
        bs.traf.ap.turnphi[:] = state["turnphi"]

        # --- Restore Guidance ---
        bs.traf.aporasas.tas[:] = state["aporasas_tas"]
        bs.traf.aporasas.alt[:] = state["aporasas_alt"]
        bs.traf.aporasas.vs[:]  = state["aporasas_vs"]
        bs.traf.aporasas.hdg[:] = state["aporasas_hdg"]

        # --- Restore Autopilot Intent ---
        bs.traf.selspd[:] = state["selspd"]
        bs.traf.swlnav[:] = state["swlnav"]
        bs.traf.swvnav[:] = state["swvnav"]
        
    def lat_lon_to_screen_coordinates (self,lat,lon,*args,**kwargs)->tuple:
        # This method should should convert the lat/lon to x/y positions on the pygame canvas
        # Since it depends on the specific environment and rendering setup, we leave it unimplemented here.
        # The user should implement this method in the subclass.
        raise NotImplementedError("This method needs to be implemented in the subclass.")
        
    def _save_env_state(self):
        state = {}
        for attr in ['wpt_reach', 'last_waypoint_distance', 'waypoint_reached', 'crashed', 'total_reward', 'total_intrusions']:
            if hasattr(self.unwrapped, attr):
                val = getattr(self.unwrapped, attr)
                if isinstance(val, list):
                    state[attr] = val.copy()
                elif isinstance(val, np.ndarray):
                    state[attr] = val.copy()
                else:
                    state[attr] = val
        return state
        
    def _restore_env_state(self, state):
        for attr, val in state.items():
            if hasattr(self.unwrapped, attr):
                if isinstance(val, list):
                    setattr(self.unwrapped, attr, val.copy())
                elif isinstance(val, np.ndarray):
                    setattr(self.unwrapped, attr, val.copy())
                else:
                    setattr(self.unwrapped, attr, val)

    def _calculate_projected_path(self,safe=False,has_waypoints=False):
        
        prev_state = self._save_traffic_state()
        prev_env_state = self._save_env_state()
        if safe:
            self.safe_action_path = []
        else:
            self.path_coordinates = []
        self._simulate_rollout(safe,has_waypoints)
        self._restore_traffic_state(prev_state)
        self._restore_env_state(prev_env_state)
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
            reward_ret = self.unwrapped._get_reward()
            if len(reward_ret) == 2:
                reward, terminated = reward_ret
                done = False
            else:
                reward, done, terminated = reward_ret
                
            if terminated or done:
                return
            
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