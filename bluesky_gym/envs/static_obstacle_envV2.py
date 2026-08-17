import numpy as np
import pygame

import bluesky as bs
import bluesky_gym.envs.common.functions as fn

import gymnasium as gym
from gymnasium import spaces

DISTANCE_MARGIN = 5 # km

REACH_REWARD = 500 # reach set waypoint
DRIFT_PENALTY = -0.01
RESTRICTED_AREA_INTRUSION_PENALTY = -500
STEP_PENALTY = -0.1 # penalty for taking too much time

INTRUSION_DISTANCE = 5 # NM

WAYPOINT_DISTANCE_MIN = 100 # KM
WAYPOINT_DISTANCE_MAX = 170 # KM

OBSTACLE_DISTANCE_MIN = 20 # KM
OBSTACLE_DISTANCE_MAX = 150 # KM

D_HEADING = 45 #degrees
D_SPEED = 20/3 # kts (check)

AC_SPD = 150 # kts
ALTITUDE = 350 # In FL

NM2KM = 1.852
MpS2Kt = 1.94384

ACTION_FREQUENCY = 10

NUM_OBSTACLES = 8
NUM_WAYPOINTS = 1

OBSTACLE_AREA_RANGE = (200, 800) # In NM^2
CENTER = (51.990426702297746, 4.376124857109851) # TU Delft AE Faculty coordinates

# minimum distance between the aircraft spawn position and the obstacle edge
SPAWN_CLEARANCE_KM = 25


MAX_DISTANCE = 350 # width of screen in km

# The line will represent where the plane will be in this many seconds if it keeps its current heading
HEADING_LENGTH_IN_SECONDS = 240


#LIDAR Basedd
RAYS = 30
DEGREE_RANGE = 180
RAYS_PER_DEGREE = RAYS/DEGREE_RANGE

class StaticObstacleEnvV2(gym.Env):
    """ 
    Static Obstacle Conflict Resolution Environment V2

    Updates from V1:
    - Dense potential-based reward for moving towards target
    - Removed speed control, fixed speed navigation
    - Higher resolution LiDAR (30 rays)
    - Re-scaled obstacles (5 obstacles, max size 300 NM^2)
    - Waypoint reset handling
    """
    
    @property
    def waypoint_distance(self):
        return self.destination_waypoint_distance

    metadata = {"render_modes": ["rgb_array","human"], "render_fps": 120}

    def __init__(self, render_mode=None, debug_lidar: bool = False):
        self.window_width = 512 # pixels
        self.window_height = 512 # pixels
        self.window_size = (self.window_width, self.window_height) # Size of the rendered environment

        self.observation_space = spaces.Dict(
            {   
                "destination_waypoint_distance": spaces.Box(-np.inf, np.inf, shape = (1,), dtype=np.float64),
                "destination_waypoint_cos_drift": spaces.Box(-np.inf, np.inf, shape = (1,), dtype=np.float64),
                "destination_waypoint_sin_drift": spaces.Box(-np.inf, np.inf, shape = (1,), dtype=np.float64),
                "lidar": spaces.Box(0.0, 1.0, shape=(RAYS,), dtype=np.float64)
            }
        )
       
        # Action space V2: Only heading change
        self.action_space = spaces.Box(-1, 1, shape=(1,), dtype=np.float64)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.debug_lidar = debug_lidar

        # initialize bluesky as non-networked simulation node
        if bs.sim is None:
            bs.init(mode='sim', detached=True)

        # set correct sim speed
        bs.stack.stack('DT 1;FF')
        
        # variables for logging
        self.total_reward = 0
        self.waypoint_reached = 0
        self.crashed = 0
        self.average_drift = np.array([])

        self.obstacle_names = []
        # keep obstacle vertices in lat/lon; segments are computed relative to
        # the current aircraft position inside `_get_obs` to support moving origin
        self.obstacle_vertices = []
        self.obstacle_segments = None
        self.lidar_ray_ends = None

        self.window = None
        self.clock = None
        
        self.last_waypoint_distance = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) 
        bs.traf.reset()

        # reset logging variables 
        self.total_reward = 0
        self.waypoint_reached = 0
        self.crashed = 0
        self.average_drift = np.array([])

        bs.traf.cre('KL001',actype="A320",acspd=AC_SPD, acalt=ALTITUDE)

        # defining screen coordinates
        # defining the reference point as the top left corner of the SQUARE screen
        # from the initial position of the aircraft which is set to be the centre of the screen
        ac_idx = bs.traf.id2idx('KL001')
        d = np.sqrt(2*(MAX_DISTANCE/2)**2) #KM
        lat_ref_point,lon_ref_point = bs.tools.geo.kwikpos(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], 315, d/NM2KM)
        
        self.screen_coords = [lat_ref_point,lon_ref_point]#[52.9, 2.6]

        self._generate_obstacles()
        self._generate_waypoint()

        ac_idx = bs.traf.id2idx('KL001')
        self.initial_wpt_qdr, initial_wpt_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], self.wpt_lat[0], self.wpt_lon[0])
        self.last_waypoint_distance = initial_wpt_dis * NM2KM
        
        bs.traf.hdg[ac_idx] = self.initial_wpt_qdr
        bs.traf.ap.trk[ac_idx] = self.initial_wpt_qdr

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def step(self, action):
        self._get_action(action)
        step_reward = 0

        for i in range(ACTION_FREQUENCY):
            bs.sim.step()
            reward, done, terminated = self._get_reward()
            step_reward += reward
            if self.render_mode == "human":
                self._render_frame()
            if terminated or done:
                observation = self._get_obs()
                self.total_reward += step_reward
                info = self._get_info()
                return observation, step_reward, done, terminated, info

        observation = self._get_obs()
        self.total_reward += step_reward
        info = self._get_info()

        return observation, step_reward, done, terminated, info

    def _generate_polygon(self, centre):
        poly_area = self.np_random.integers(OBSTACLE_AREA_RANGE[0]*2, OBSTACLE_AREA_RANGE[1])
        R = np.sqrt(poly_area/ np.pi)
        p = [fn.random_point_on_circle(R,self.np_random) for _ in range(3)] # 3 random points to start building the polygon
        p = fn.sort_points_clockwise(p)
        p_area = fn.polygon_area(p)
        
        while p_area < OBSTACLE_AREA_RANGE[0]:
            p.append(fn.random_point_on_circle(R,self.np_random))
            p = fn.sort_points_clockwise(p)
            p_area = fn.polygon_area(p)
        
        p = [fn.nm_to_latlong(centre, point) for point in p] # Convert to lat/long coordinateS
        return p_area, p, R
    
    def _generate_obstacles(self):
        # delete existing obstacles from previous episode in BlueSky
        for name in self.obstacle_names:
            bs.tools.areafilter.deleteArea(name)

        self.obstacle_names = []
        self.obstacle_radius = []
        self.obstacle_vertices = []

        ac_idx = bs.traf.id2idx('KL001')
        spawn_lat = bs.traf.lat[ac_idx]
        spawn_lon = bs.traf.lon[ac_idx]

        self.obstacle_centre_lat = []
        self.obstacle_centre_lon = []

        for i in range(NUM_OBSTACLES):
            accepted = False
            loop_counter = 0
            while not accepted:
                loop_counter += 1
                obstacle_dis_from_reference = self.np_random.integers(OBSTACLE_DISTANCE_MIN, OBSTACLE_DISTANCE_MAX)
                obstacle_hdg_from_reference = self.np_random.integers(0, 360)
                centre_lat, centre_lon = fn.get_point_at_distance(spawn_lat, spawn_lon, obstacle_dis_from_reference, obstacle_hdg_from_reference)
                centre_obst = (centre_lat, centre_lon)
                _, p, R = self._generate_polygon(centre_obst)

                centre_dist_nm = bs.tools.geo.kwikqdrdist(spawn_lat, spawn_lon, centre_obst[0], centre_obst[1])[1]
                centre_dist_km = centre_dist_nm * NM2KM
                obstacle_edge_clearance_km = centre_dist_km - (R * NM2KM)

                if obstacle_edge_clearance_km < SPAWN_CLEARANCE_KM:
                    continue

                points = [coord for point in p for coord in point] # Flatten the list of points
                poly_name = 'restricted_area_' + str(i+1)
                bs.tools.areafilter.defineArea(poly_name, 'POLY', points)
                self.obstacle_names.append(poly_name)
                self.obstacle_centre_lat.append(centre_lat)
                self.obstacle_centre_lon.append(centre_lon)

                obstacle_vertices_coordinates = []
                for k in range(0,len(points),2):
                    obstacle_vertices_coordinates.append([points[k], points[k+1]])
                
                self.obstacle_vertices.append(obstacle_vertices_coordinates)
                self.obstacle_radius.append(R)
                accepted = True

                if loop_counter > 1000:
                    raise Exception("Unable to spawn obstacles with sufficient clearance from the aircraft start position.")

        # segments will be computed on-the-fly in `_get_obs` relative to AC
        self.obstacle_segments = None

    def _generate_waypoint(self, acid = 'KL001'):
        self.wpt_lat = []
        self.wpt_lon = []
        self.wpt_reach = []

        ac_idx = bs.traf.id2idx(acid)
        check_inside_var = True
        loop_counter = 0
        while check_inside_var:
            loop_counter += 1
            wpt_dis_init = self.np_random.integers(WAYPOINT_DISTANCE_MIN, WAYPOINT_DISTANCE_MAX)
            wpt_hdg_init = self.np_random.integers(0, 360)
            wpt_lat, wpt_lon = fn.get_point_at_distance(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], wpt_dis_init, wpt_hdg_init)

            inside_temp = []
            for j in range(NUM_OBSTACLES):
                dist_nm = bs.tools.geo.kwikqdrdist(wpt_lat, wpt_lon, self.obstacle_centre_lat[j], self.obstacle_centre_lon[j])[1]
                dist_km = dist_nm * NM2KM
                clearance = dist_km - (self.obstacle_radius[j] * NM2KM)
                # Ensure waypoint is at least 25 km away from the edge of any obstacle
                if clearance < 25.0:
                    inside_temp.append(True)
                else:
                    inside_temp.append(bs.tools.areafilter.checkInside(self.obstacle_names[j], np.array([wpt_lat]), np.array([wpt_lon]), np.array([bs.traf.alt[ac_idx]]))[0])
            
            check_inside_var = any(x == True for x in inside_temp)
      
            if loop_counter > 1000:
                raise Exception("No waypoints can be generated outside the obstacles. Check the parameters of the obstacles in the definition of the scenario.")

        self.wpt_lat.append(wpt_lat)
        self.wpt_lon.append(wpt_lon)
        self.wpt_reach.append(0)

    def _generate_coordinates_centre_obstacles(self, acid = 'KL001', num_obstacles = NUM_OBSTACLES):
        self.obstacle_centre_lat = []
        self.obstacle_centre_lon = []
        
        for i in range(num_obstacles):
            obstacle_dis_from_reference = self.np_random.integers(OBSTACLE_DISTANCE_MIN, OBSTACLE_DISTANCE_MAX)
            obstacle_hdg_from_reference = self.np_random.integers(0, 360)
            ac_idx = bs.traf.id2idx(acid)

            obstacle_centre_lat, obstacle_centre_lon = fn.get_point_at_distance(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], obstacle_dis_from_reference, obstacle_hdg_from_reference)    
            self.obstacle_centre_lat.append(obstacle_centre_lat)
            self.obstacle_centre_lon.append(obstacle_centre_lon)

    def _get_obs(self):
        ac_idx = bs.traf.id2idx('KL001')

        self.destination_waypoint_distance = []
        self.wpt_qdr = []
        self.destination_waypoint_cos_drift = []
        self.destination_waypoint_sin_drift = []
        self.destination_waypoint_drift = []

        self.obstacle_centre_distance = []
        self.obstacle_centre_cos_bearing = []
        self.obstacle_centre_sin_bearing = []
            
        self.ac_hdg = bs.traf.hdg[ac_idx]
        self.ac_tas = bs.traf.tas[ac_idx]

        wpt_qdr, wpt_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], self.wpt_lat[0], self.wpt_lon[0])
    
        self.destination_waypoint_distance.append(wpt_dis * NM2KM)
        self.wpt_qdr.append(wpt_qdr)

        drift = self.ac_hdg - wpt_qdr
        drift = fn.bound_angle_positive_negative_180(drift)

        self.destination_waypoint_drift.append(drift)
        self.destination_waypoint_cos_drift.append(np.cos(np.deg2rad(drift)))
        self.destination_waypoint_sin_drift.append(np.sin(np.deg2rad(drift)))
        
        for obs_idx in range(NUM_OBSTACLES):
            obs_centre_qdr, obs_centre_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], self.obstacle_centre_lat[obs_idx], self.obstacle_centre_lon[obs_idx])
            obs_centre_dis = obs_centre_dis * NM2KM #KM        

            bearing = self.ac_hdg - obs_centre_qdr
            bearing = fn.bound_angle_positive_negative_180(bearing)

            self.obstacle_centre_distance.append(obs_centre_dis)
            self.obstacle_centre_cos_bearing.append(np.cos(np.deg2rad(bearing)))
            self.obstacle_centre_sin_bearing.append(np.sin(np.deg2rad(bearing)))

        # --- LIDAR computation: build rays and test intersections vectorized ---
        obs_segs_list = []
        for vertices in self.obstacle_vertices:
            pts_nm = [fn.latlong_to_nm(np.array([bs.traf.lat[ac_idx], bs.traf.lon[ac_idx]]), np.array([v[0], v[1]])) for v in vertices]
            if len(pts_nm) >= 2:
                segs = fn.polygon_to_segments(np.array(pts_nm))
                obs_segs_list.append(segs)

        if len(obs_segs_list) > 0:
            all_obs_segs = np.vstack(obs_segs_list)
        else:
            all_obs_segs = np.zeros((0,4), dtype=float)

        ray_angles = (np.linspace(-DEGREE_RANGE/2, DEGREE_RANGE/2, RAYS) + self.ac_hdg) % 360
        ray_len_nm = MAX_DISTANCE / NM2KM
        ang_rad = np.deg2rad(ray_angles)
        ray_ends_x = ray_len_nm * np.cos(ang_rad)
        ray_ends_y = ray_len_nm * np.sin(ang_rad)
        rays = np.column_stack((np.zeros(RAYS), np.zeros(RAYS), ray_ends_x, ray_ends_y))

        if all_obs_segs.shape[0] == 0:
            lidar_ranges = np.ones(RAYS, dtype=float)
            t_min = np.full(RAYS, np.nan)
        else:
            t_mat, _, mask = fn.segments_intersection_params(rays, all_obs_segs)
            t_mat[~mask] = np.nan
            t_min = np.nanmin(t_mat, axis=1)
            lidar_ranges = np.where(np.isnan(t_min), 1.0, np.clip(t_min * (ray_len_nm * NM2KM) / MAX_DISTANCE, 0.0, 1.0))

        effective_t = np.where(np.isnan(t_min), 1.0, t_min)
        distances_nm = effective_t * ray_len_nm
        distances_km = distances_nm * NM2KM
        ray_angles = (np.linspace(-DEGREE_RANGE/2, DEGREE_RANGE/2, RAYS) + self.ac_hdg) % 360
        ends_lat = []
        ends_lon = []
        for ang, d_km in zip(ray_angles, distances_km):
            lat_e, lon_e = fn.get_point_at_distance(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], d_km, ang)
            ends_lat.append(lat_e)
            ends_lon.append(lon_e)
        self.lidar_ray_ends = (np.array(ends_lat), np.array(ends_lon))


        observation = {
                "destination_waypoint_distance": np.array(self.destination_waypoint_distance)/WAYPOINT_DISTANCE_MAX,
                "destination_waypoint_cos_drift": np.array(self.destination_waypoint_cos_drift),
                "destination_waypoint_sin_drift": np.array(self.destination_waypoint_sin_drift),
                "lidar": np.array(lidar_ranges)
            }

        return observation

    def _get_info(self):
        return {
            'total_reward': self.total_reward,
            'waypoint_reached': self.waypoint_reached,
            'crashed': self.crashed,
            'average_drift': self.average_drift.mean() if len(self.average_drift) > 0 else 0
        }

    def _get_reward(self):
        ac_idx = bs.traf.id2idx('KL001')
        wpt_qdr, wpt_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], self.wpt_lat[0], self.wpt_lon[0])
        current_distance = wpt_dis * NM2KM
        
        # 1. Waypoint Reach
        reach_reward = 0
        if current_distance < DISTANCE_MARGIN and self.wpt_reach[0] != 1:
            self.waypoint_reached = 1
            self.wpt_reach[0] = 1
            reach_reward = REACH_REWARD

        # 2. Drift Penalty
        drift = fn.bound_angle_positive_negative_180(bs.traf.hdg[ac_idx] - wpt_qdr)
        drift = abs(np.deg2rad(drift))
        self.average_drift = np.append(self.average_drift, drift)
        drift_reward = drift * DRIFT_PENALTY

        # 3. Intrusion
        intrusion_reward = 0
        intrusion_terminate = 0
        for obs_idx in range(NUM_OBSTACLES):
            if bs.tools.areafilter.checkInside(self.obstacle_names[obs_idx], np.array([bs.traf.lat[ac_idx]]), np.array([bs.traf.lon[ac_idx]]), np.array([bs.traf.alt[ac_idx]])):
                intrusion_reward += RESTRICTED_AREA_INTRUSION_PENALTY
                self.crashed = 1
                intrusion_terminate = 1
                break # Only penalize once

        # 4. Dense Distance Reward
        distance_reward = 0
        if self.last_waypoint_distance is not None:
            distance_reduction = self.last_waypoint_distance - current_distance
            distance_reward = distance_reduction * 1.0 # 1 reward point per km moved closer
        self.last_waypoint_distance = current_distance
        
        # 5. Out of bounds check
        out_of_bounds_terminate = False
        out_of_bounds_penalty = 0
        if current_distance > WAYPOINT_DISTANCE_MAX * 1.5:
            out_of_bounds_terminate = True
            out_of_bounds_penalty = -100
            
        total_reward = reach_reward + drift_reward + intrusion_reward + distance_reward + out_of_bounds_penalty + STEP_PENALTY
        
        done = 0
        if self.wpt_reach[0] == 1 or intrusion_terminate or out_of_bounds_terminate:
            done = 1

        return total_reward, done, False

    def _get_action(self,action):
        dh = action[0] * D_HEADING
        heading_new = fn.bound_angle_positive_negative_180(bs.traf.hdg[bs.traf.id2idx('KL001')] + dh)
        bs.stack.stack(f"HDG {'KL001'} {heading_new}")

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        screen_coords = self.screen_coords

        canvas = pygame.Surface(self.window_size)
        canvas.fill((135,206,235))

        px_per_km = self.window_width/MAX_DISTANCE

        # draw ownship
        ac_idx = bs.traf.id2idx('KL001')
        ac_length = 8
        heading_end_x = ((np.sin(np.deg2rad(bs.traf.hdg[ac_idx])) * ac_length)/MAX_DISTANCE)*self.window_width
        heading_end_y = ((np.cos(np.deg2rad(bs.traf.hdg[ac_idx])) * ac_length)/MAX_DISTANCE)*self.window_width

        qdr, dis = bs.tools.geo.kwikqdrdist(screen_coords[0], screen_coords[1], bs.traf.lat[ac_idx], bs.traf.lon[ac_idx])
        dis = dis*NM2KM
        x_actor = ((np.sin(np.deg2rad(qdr))*dis)/MAX_DISTANCE)*self.window_width
        y_actor = ((-np.cos(np.deg2rad(qdr))*dis)/MAX_DISTANCE)*self.window_width
        pygame.draw.line(canvas,
            (235, 52, 52),
            (x_actor, y_actor),
            (x_actor+heading_end_x, y_actor-heading_end_y),
            width = 5
        )

        # draw heading line with variable length depending on seconds into the future
        PX2KM = self.window_width/MAX_DISTANCE
        ac_spd = bs.traf.cas[ac_idx] # m/s
        heading_length_km = ac_spd/1000 * HEADING_LENGTH_IN_SECONDS
        heading_length_px = heading_length_km * PX2KM
        heading_end_x = ((np.sin(np.deg2rad(bs.traf.hdg[ac_idx])) * heading_length_px))
        heading_end_y = ((np.cos(np.deg2rad(bs.traf.hdg[ac_idx])) * heading_length_px))
        pygame.draw.line(canvas,
            (0,0,0),
            (x_actor,y_actor),
            (x_actor+heading_end_x, y_actor-heading_end_y),
            width = 1
        )

        # draw lidar rays if debug enabled
        if self.debug_lidar and self.lidar_ray_ends is not None:
            lat_ends, lon_ends = self.lidar_ray_ends
            for lat_e, lon_e in zip(lat_ends, lon_ends):
                qdr_e, dis_e = bs.tools.geo.kwikqdrdist(screen_coords[0], screen_coords[1], lat_e, lon_e)
                dis_e = dis_e * NM2KM
                x_e = (np.sin(np.deg2rad(qdr_e)) * dis_e)/MAX_DISTANCE*self.window_width
                y_e = (-np.cos(np.deg2rad(qdr_e)) * dis_e)/MAX_DISTANCE*self.window_width
                pygame.draw.line(canvas, (0,255,0), (x_actor, y_actor), (x_e, y_e), width=1)

        # draw obstacles
        for vertices in self.obstacle_vertices:
            points = []
            for coord in vertices:
                lat_ref = coord[0]
                lon_ref = coord[1]
                qdr, dis = bs.tools.geo.kwikqdrdist(screen_coords[0], screen_coords[1], lat_ref, lon_ref)
                dis = dis*NM2KM
                x_ref = (np.sin(np.deg2rad(qdr))*dis)/MAX_DISTANCE*self.window_width
                y_ref = (-np.cos(np.deg2rad(qdr))*dis)/MAX_DISTANCE*self.window_width
                points.append((x_ref, y_ref))
            pygame.draw.polygon(canvas,
                (0,0,0), points
            )

        # draw target waypoint
        indx = 0
        for lat, lon, reach in zip(self.wpt_lat, self.wpt_lon, self.wpt_reach):
            indx += 1
            qdr, dis = bs.tools.geo.kwikqdrdist(screen_coords[0], screen_coords[1], lat, lon)

            circle_x = ((np.sin(np.deg2rad(qdr)) * dis * NM2KM)/MAX_DISTANCE)*self.window_width
            circle_y = (-(np.cos(np.deg2rad(qdr)) * dis * NM2KM)/MAX_DISTANCE)*self.window_width

            color = (255,255,255)

            pygame.draw.circle(
                canvas, 
                color,
                (circle_x,circle_y),
                radius = 4,
                width = 0
            )
            
            pygame.draw.circle(
                canvas, 
                color,
                (circle_x,circle_y),
                radius = (DISTANCE_MARGIN/MAX_DISTANCE)*self.window_width,
                width = 2
            )

        self.window.blit(canvas, canvas.get_rect())
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        bs.stack.stack('quit')
