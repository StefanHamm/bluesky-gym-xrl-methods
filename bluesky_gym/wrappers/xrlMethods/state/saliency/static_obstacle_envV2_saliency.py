from bluesky_gym.utils.constants import HEADING_LENGTH_IN_SECONDS
import gymnasium as gym
import numpy as np
import pygame
from bluesky_gym.envs.static_obstacle_envV2 import ACTION_FREQUENCY, NM2KM, AC_SPD, WAYPOINT_DISTANCE_MAX, RAYS, DEGREE_RANGE, MAX_DISTANCE
from bluesky_gym.wrappers.xrlMethods.state.general_saliency import SaliencyMapV1Wrapper
import bluesky as bs

import bluesky_gym.envs.common.functions as fn
import os

class SaliencyStaticObstacleControl(SaliencyMapV1Wrapper):
    def __init__(self, env, safe_vals=None, debug=False, export_gifs_path=None, fps=5, color_mode="clipped", plot_action_path=False, plot_safe_path=False, model=None, xrl_rendering=True):
        super().__init__(env, safe_vals, debug, export_gifs_path, fps, color_mode, model, xrl_rendering=xrl_rendering)
        self.action_frequency = ACTION_FREQUENCY
        self.plot_action_path = plot_action_path
        self.plot_safe_path = plot_safe_path
        
        from bluesky_gym.envs.static_obstacle_envV2 import INTRUSION_DISTANCE
        self.distance_margin = INTRUSION_DISTANCE * NM2KM

    def reset(self, seed=None, options=None):
        obs, inf = super().reset(seed=seed, options=options)
        self.episode_counter += 1
        self.step_counter = 0

        if self.export_gifs_path is not None:
            self.episode_frames_path = os.path.join(self.frames_path, f"episode_{self.episode_counter}")
            os.makedirs(self.episode_frames_path, exist_ok=True)

        if self.plot_action_path and self.model is not None:
            self._calculate_projected_path(safe=False, has_waypoints=True)

        if self.render_mode == "human":
            self._render_frame()

        return obs, inf

    def step(self, action, shap_values=None, examplePlane=None):
        if not hasattr(self, 'path_update_counter'):
            self.path_update_counter = 0
            
        if self.plot_action_path and self.model is not None:
            if self.path_update_counter % 10 == 0:
                self._calculate_projected_path(safe=False, has_waypoints=True)
            
        if self.plot_safe_path and self.model is not None and self.safe_vals is not None:
            if self.path_update_counter % 10 == 0:
                 self._calculate_projected_path(safe=True, has_waypoints=True)

        self.path_update_counter += 1
        self.unwrapped._get_action(action)
        self.last_action = action

        self.step_counter += 1
        self.frame_saved = False

        for i in range(self.action_frequency):
            if self.render_mode == "human":
                self._render_frame(shap_values=shap_values)
            bs.sim.step()

        observation = self.unwrapped._get_obs()
        reward, done, terminated = self.unwrapped._get_reward()
        info = self.unwrapped._get_info()

        if terminated or done:
            self.export_episode_gif()

        return observation, reward, done or terminated, False, info

    def lat_lon_to_screen_coordinates(self, lat, lon):
        screen_coords = self.unwrapped.screen_coords
        qdr, dis = bs.tools.geo.kwikqdrdist(screen_coords[0], screen_coords[1], lat, lon)
        dis = dis * NM2KM
        x_pos = ((np.sin(np.deg2rad(qdr))*dis)/MAX_DISTANCE)*self.unwrapped.window_width
        y_pos = ((-np.cos(np.deg2rad(qdr))*dis)/MAX_DISTANCE)*self.unwrapped.window_width
        return x_pos, y_pos

    def _render_frame(self, shap_values=None):
        self._pre_render()
        ac_idx = bs.traf.id2idx('KL001')
        max_distance = MAX_DISTANCE
        canvas = pygame.Surface(self.unwrapped.window_size)
        canvas.fill((135,206,235))

        if self.plot_action_path and self.model is not None:
            self._draw_path(canvas, (255,0,0), self.path_coordinates, True)
        if self.plot_safe_path and self.model is not None and self.safe_vals is not None:
            self._draw_path(canvas, (255,0,255), self.safe_action_path)

        px_per_km = self.unwrapped.window_width / max_distance
        screen_coords = self.unwrapped.screen_coords

        # Map LiDAR SHAP values to Obstacles
        num_obstacles = len(self.unwrapped.obstacle_names)
        obstacle_shap_sums = [0.0] * num_obstacles
        
        if shap_values is not None and self.xrl_rendering:
            ray_angles_relative = np.linspace(-DEGREE_RANGE/2, DEGREE_RANGE/2, RAYS)
            own_hdg = bs.traf.hdg[ac_idx]
            lidar_ranges = self.unwrapped._get_obs()["lidar"]
            max_dist_nm = max_distance / NM2KM
            
            # 1. Accumulate SHAP sums for each obstacle
            ray_draw_data = []
            for i in range(RAYS):
                ray_val = lidar_ranges[i]
                if ray_val < 1.0: # obstacle detected
                    abs_angle = (own_hdg + ray_angles_relative[i]) % 360
                    dist_km = ray_val * max_dist_nm * NM2KM
                    lat_e, lon_e = fn.get_point_at_distance(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], dist_km, abs_angle)
                    
                    min_dist = float('inf')
                    closest_obs = -1
                    for j in range(num_obstacles):
                        obs_lat = self.unwrapped.obstacle_centre_lat[j]
                        obs_lon = self.unwrapped.obstacle_centre_lon[j]
                        _, dist_to_center = bs.tools.geo.kwikqdrdist(lat_e, lon_e, obs_lat, obs_lon)
                        if dist_to_center < min_dist:
                            min_dist = dist_to_center
                            closest_obs = j
                            
                    if closest_obs != -1:
                        shap_idx = 3 + i
                        obstacle_shap_sums[closest_obs] += shap_values.values[0][shap_idx]
                        ray_draw_data.append((lat_e, lon_e, closest_obs))

            # 2. Compute the true maximum SHAP value across logically grouped features
            grouped_shaps = [
                abs(shap_values.values[0][0]), # distance
                abs(shap_values.values[0][1]), # cos drift
                abs(shap_values.values[0][2])  # sin drift
            ] + [abs(s) for s in obstacle_shap_sums]
            true_max_shap = max(grouped_shaps) if grouped_shaps else np.max(np.abs(shap_values.values))

            # 3. Draw LiDAR Rays with synchronized scaling
            if self.unwrapped.debug_lidar:
                for lat_e, lon_e, closest_obs in ray_draw_data:
                    obs_shap = obstacle_shap_sums[closest_obs]
                    ray_color = self._get_saliency_color(obs_shap, true_max_shap, shap_values.base_values[0][0])
                    
                    qdr_e, dis_e = bs.tools.geo.kwikqdrdist(screen_coords[0], screen_coords[1], lat_e, lon_e)
                    dis_e = dis_e * NM2KM
                    x_e = ((np.sin(np.deg2rad(qdr_e))*dis_e)/max_distance)*self.unwrapped.window_width
                    y_e = ((-np.cos(np.deg2rad(qdr_e))*dis_e)/max_distance)*self.unwrapped.window_width
                    
                    qdr_own, dis_own = bs.tools.geo.kwikqdrdist(screen_coords[0], screen_coords[1], bs.traf.lat[ac_idx], bs.traf.lon[ac_idx])
                    dis_own = dis_own * NM2KM
                    own_x = (np.sin(np.deg2rad(qdr_own))*dis_own)/max_distance*self.unwrapped.window_width
                    own_y = (-np.cos(np.deg2rad(qdr_own))*dis_own)/max_distance*self.unwrapped.window_width
                    
                    pygame.draw.line(canvas, ray_color, (own_x, own_y), (x_e, y_e), width=3)

        # Draw obstacles
        for j, vertices in enumerate(self.unwrapped.obstacle_vertices):
            points = []
            for coord in vertices:
                lat_ref = coord[0]
                lon_ref = coord[1]
                qdr, dis = bs.tools.geo.kwikqdrdist(screen_coords[0], screen_coords[1], lat_ref, lon_ref)
                dis = dis * NM2KM
                x_ref = (np.sin(np.deg2rad(qdr))*dis)/MAX_DISTANCE*self.unwrapped.window_width
                y_ref = (-np.cos(np.deg2rad(qdr))*dis)/MAX_DISTANCE*self.unwrapped.window_width
                points.append((x_ref, y_ref))
                
            if shap_values is not None and self.xrl_rendering:
                obs_shap = obstacle_shap_sums[j]
                if obs_shap != 0.0:
                    color = self._get_saliency_color(obs_shap, true_max_shap, shap_values.base_values[0][0])
                else:
                    color = (80,80,80) # Grey for unobserved obstacles
            else:
                color = (0,0,0) # Default black if no XRL
                
            pygame.draw.polygon(canvas, color, points)

        # Draw Target Waypoint Line (Efficiency Pull)
        if shap_values is not None and self.xrl_rendering:
            waypoint_shap_sum = shap_values.values[0][0] + shap_values.values[0][1] + shap_values.values[0][2]
            base_val = shap_values.base_values[0][0]
            max_abs_shap = np.max(np.abs(shap_values.values))
            waypoint_color = self._get_saliency_color(waypoint_shap_sum, max_abs_shap, base_val)
            
            waypoint_qdr = self.unwrapped.wpt_qdr[0]
            waypoint_dis = self.unwrapped.waypoint_distance[0]
            
            # Use screen coords to find waypoint position
            wpt_lat, wpt_lon = self.unwrapped.wpt_lat[0], self.unwrapped.wpt_lon[0]
            qdr, dis = bs.tools.geo.kwikqdrdist(screen_coords[0], screen_coords[1], wpt_lat, wpt_lon)
            dis = dis * NM2KM
            circle_x = (np.sin(np.deg2rad(qdr)) * dis)/max_distance*self.unwrapped.window_width
            circle_y = (-np.cos(np.deg2rad(qdr)) * dis)/max_distance*self.unwrapped.window_width
            
            # Ownship position relative to screen coords
            qdr_own, dis_own = bs.tools.geo.kwikqdrdist(screen_coords[0], screen_coords[1], bs.traf.lat[ac_idx], bs.traf.lon[ac_idx])
            dis_own = dis_own * NM2KM
            own_x = (np.sin(np.deg2rad(qdr_own))*dis_own)/max_distance*self.unwrapped.window_width
            own_y = (-np.cos(np.deg2rad(qdr_own))*dis_own)/max_distance*self.unwrapped.window_width

            pygame.draw.line(canvas,
                waypoint_color,
                (own_x, own_y),
                (circle_x, circle_y),
                width=6
            )

        # Draw waypoints standard
        for lat, lon, reach in zip(self.unwrapped.wpt_lat, self.unwrapped.wpt_lon, self.unwrapped.wpt_reach):
            qdr, dis = bs.tools.geo.kwikqdrdist(screen_coords[0], screen_coords[1], lat, lon)
            dis = dis * NM2KM
            circle_x = (np.sin(np.deg2rad(qdr)) * dis)/max_distance*self.unwrapped.window_width
            circle_y = (-np.cos(np.deg2rad(qdr)) * dis)/max_distance*self.unwrapped.window_width
            
            color = (155,155,155) if reach else (255,255,255)
            pygame.draw.circle(canvas, color, (circle_x, circle_y), radius=4, width=0)
            
            from bluesky_gym.envs.static_obstacle_envV2 import INTRUSION_DISTANCE
            margin = INTRUSION_DISTANCE * NM2KM
            pygame.draw.circle(canvas, color, (circle_x, circle_y), radius=(margin/max_distance)*self.unwrapped.window_width, width=2)

        # draw ownship
        qdr_own, dis_own = bs.tools.geo.kwikqdrdist(screen_coords[0], screen_coords[1], bs.traf.lat[ac_idx], bs.traf.lon[ac_idx])
        dis_own = dis_own * NM2KM
        own_x = (np.sin(np.deg2rad(qdr_own))*dis_own)/max_distance*self.unwrapped.window_width
        own_y = (-np.cos(np.deg2rad(qdr_own))*dis_own)/max_distance*self.unwrapped.window_width
        x_actor = own_x
        y_actor = own_y

        ac_length = 8
        heading_end_x = np.sin(np.deg2rad(bs.traf.hdg[ac_idx])) * ac_length * px_per_km
        heading_end_y = np.cos(np.deg2rad(bs.traf.hdg[ac_idx])) * ac_length * px_per_km
        pygame.draw.line(canvas,
            (0,0,0),
            (x_actor - heading_end_x/2, y_actor + heading_end_y/2),
            (x_actor + heading_end_x/2, y_actor - heading_end_y/2),
            width=4
        )
        
        # draw heading line (projected position)
        ac_spd = bs.traf.cas[ac_idx]  # [m/s]
        heading_length_km = (ac_spd * HEADING_LENGTH_IN_SECONDS) / 1000.0
        heading_end_x = np.sin(np.deg2rad(bs.traf.hdg[ac_idx])) * heading_length_km * px_per_km
        heading_end_y = np.cos(np.deg2rad(bs.traf.hdg[ac_idx])) * heading_length_km * px_per_km
        pygame.draw.line(canvas,
            (0,0,0),
            (x_actor, y_actor),
            (x_actor + heading_end_x, y_actor - heading_end_y),
            width=1
        )

        # Draw legend for SHAP influence
        legend_x = 30
        legend_y = self.unwrapped.window_size[1] - 80
        legend_width = 200
        legend_height = 20

        if shap_values is not None and self.xrl_rendering:
            shap_sums = [float(np.sum(shap_values.values))]
            if self.DEBUG:
                action_taken = [float(self.last_action)]
                baseline_value = shap_values.base_values[0]
                self._draw_debug_menue(canvas, legend_x, legend_y, action_taken, shap_sums, baseline_value)

            self._draw_shap_bar(canvas, shap_sums[0], legend_x, legend_y, legend_width, legend_height, "horizontal", "Right", "Left", "Overall Turn Influence")

        if not self.frame_saved and self.export_gifs_path is not None and shap_values is not None:
            self._create_shap_row(shap_values.values[0])

        self._post_render(canvas)
