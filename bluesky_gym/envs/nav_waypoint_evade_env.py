import numpy as np
import pygame
import matplotlib.pyplot as plt
import bluesky as bs
from bluesky_gym.envs.common.screen_dummy import ScreenDummy
import bluesky_gym.envs.common.functions as fn
import math
import matplotlib.path
from scipy import ndimage

import gymnasium as gym
from gymnasium import spaces
import networkx as nx

SENSOR_RANGE = 200 #km


AIRWAY_WIDTH = 8 # NM

DEBUG = True


# REWARDS
REACH_REWARD = 5
DRIFT_PENALTY = -0.5
ALTITUDE_PENALTY = -1.5 # altitude higher penalty than drift since more fuel cost
INTRUSION_PENALTY = -4
CORRIDOR_LEAVE_PENALTY = -1
OBSTACLE_PENALTY = -30
CRASH_PENALTY = -100


# INTRUDER AND WAPOINT CONFIG
NUM_INTRUDERS = 5
NUM_WAYPOINTS = 1
INTRUSION_DISTANCE = 5 # NM
CRASH_DISTANCE_HORIZONTAL = 0.1 #NM
CRASH_DISTANCE_VERTICAL = 50 # m

MIN_ROUTE_LENGTH = 9 #


AC_SPD = 150 #m/s

NM2KM = 1.852
KT2MPS = 0.514444

ACTION_FREQUENCY = 1

FT_TO_M = 0.3048
AC_TYPE = "A320"

FLIGHT_LEVEL = 340 #FL340
FLIGHT_LEVEL_FT = FLIGHT_LEVEL * 100
FLIGHT_LEVEL_M = FLIGHT_LEVEL_FT * FT_TO_M 

VERTICAL_SEPARATION_IN_FT = 1000
VERTICAL_SEPARATION_IN_M = VERTICAL_SEPARATION_IN_FT * FT_TO_M

WAYPOINT_REACH_DISTANCE = 5 # NM

INTRUDER_ALT_SPANRANGE_IN_1000FT = 2 # this means the intruder can be between FL320 and FL360 if the agent is at FL340, this allows for more realistic encounters where the intruder is not always at the same altitude as the agent.

# NAVPOINTS
EDGES_PATH = "data/edges.csv"
VERTICES_PATH = "data/vertices.csv"

#ACTION Space
#for heading change its discrete with upto ALTITUDE_STEPS in each direction
#the step height is set as D_ALTITUDE
ALTITUDE_STEPS = 4
D_ALTITUDE = VERTICAL_SEPARATION_IN_M/2 # the alitude 5 FL e.g. FL430 FL435 etc.
CLIMB_RATE = 5 # m/s
DECENT_RATE = -5 # m/s
ALT_REACH_DISTANCE = 50 #m how close it needs to be to accept new commands


D_HEADING = 45 #degree action of 1 changes heading by 45 degree


# --- ATC REALISM COLOR PALETTE ---
COLORS = {
    "BACKGROUND": (20, 24, 28),         # Deep Radar Grey/Black
    "VERTICAL_BG": (30, 35, 40),        # UI panels
    "TEXT": (200, 220, 220),            # Off-white/Pale Cyan
    "GRID_LINES": (60, 70, 80),         # Subtle grid lines
    
    # Aircraft & Traffic
    "OWNSHIP": (100, 255, 218),         # Cyan/Teal
    "INTRUDER_SAFE": (255, 191, 0),     # Amber/Orange
    "INTRUDER_CONFLICT": (255, 50, 50), # Bright Red
    "TRAIL": (100, 100, 100),           # Grey for trails
    
    # Navigation
    "AIRWAY": (46, 139, 87),            # SeaGreen (Generic)
    "AIRWAY_ACTIVE": (0, 255, 40),     # Strong green (Active Route)
    "WAYPOINT": (169, 169, 169),        # Dark Grey
    "WAYPOINT_ACTIVE": (255, 255, 255), # White
    
    # --- NEW: Airway Specifics ---
    "AIRWAY_CORRIDOR": (35, 55, 45),    # Dark background width (Safe Airspace)
    "AIRWAY_CENTER": (80, 120, 100),    # Sharp foreground line (Centerline)
    
    # Debug/Logic (Bisectors)
    "BISECTOR_PASSED": (60, 60, 80),    # Faded Blue-Grey
    "BISECTOR_ACTIVE": (0, 191, 255),   # Deep Sky Blue
    "BISECTOR_FUTURE": (70, 130, 180),  # Steel Blue
    
    # UI Elements
    "SCALE_LINE": (255, 255, 255)
}


def load_graph(vetices_path=VERTICES_PATH, edges_path=EDGES_PATH):
    graph = nx.Graph()
    with open(vetices_path, 'r') as f:
        f.readline() # skip first line
        for line in f:
            parts = line.strip().split(',')
            node_id = str(parts[0])
            lat = float(parts[1])
            lon = float(parts[2])
            alitude = float(parts[3])
            is_ariport = int(parts[4])
            if not is_ariport:
                graph.add_node(node_id, lat=lat, lon=lon)
    # then load the edges
    with open(edges_path, 'r') as f:
        #skip first line
        f.readline()
        for line in f:
            parts = line.strip().split(',')
            node1 = str(parts[0])
            node2 = str(parts[1])
            distance = float(parts[2])
            # Only add edge if both nodes exist
            if node1 in graph.nodes and node2 in graph.nodes:
                graph.add_edge(node1, node2, weight=distance)
    return graph


class NavWaypointEvadeEnv(gym.Env):
    """Navpoint gymnasium creating random navpoint encounters

    Args:
        gym (_type_): _description_
    """
    metadata = {"render_modes": ["rgb_array","human"], "render_fps": 20}
    
    
    
    def __init__(self, render_mode=None, window_width=500,window_height=500, stencil_radius_in_km = 100, show_altitude_in_rendering=True,workdir=None,plot_all_points = True):
        super().__init__()
        
        # load the graph
        # first load the vertices
        self.graph = load_graph(VERTICES_PATH, EDGES_PATH)
        self.window_width = window_width
        self.window_height = window_height
        self.plot_all_points = plot_all_points
        if show_altitude_in_rendering:
            self.window_height +=200
            self.show_altitude_in_rendering = True
        
        
        self.window_size = (self.window_width, self.window_height)
        self.window = None
        self.clock = None
        self.agent_nav_path = None
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        
        self.observation_space = spaces.Dict(
            {
                "intruder_distance": spaces.Box(-np.inf, np.inf, shape = (NUM_INTRUDERS,), dtype=np.float64),
                "cos_difference_pos": spaces.Box(-np.inf, np.inf, shape = (NUM_INTRUDERS,), dtype=np.float64),
                "sin_difference_pos": spaces.Box(-np.inf, np.inf, shape = (NUM_INTRUDERS,), dtype=np.float64),
                "z_difference_pos": spaces.Box(-np.inf, np.inf, shape=(NUM_INTRUDERS,), dtype=np.float64),
                "x_difference_speed": spaces.Box(-np.inf, np.inf, shape = (NUM_INTRUDERS,), dtype=np.float64),
                "y_difference_speed": spaces.Box(-np.inf, np.inf, shape = (NUM_INTRUDERS,), dtype=np.float64),

                "waypoint_distance": spaces.Box(-np.inf, np.inf, shape = (3,), dtype=np.float64), # always has previous current and next waypoint this allows it to learn the drift in airway.
                "waypoint_cos_pos": spaces.Box(-np.inf, np.inf, shape = (3,), dtype=np.float64),
                "waypoint_sin_pos": spaces.Box(-np.inf, np.inf, shape = (3,), dtype=np.float64),
                "waypoint_mask": spaces.Box(0, 1, shape = (3,), dtype=np.float64), # this indicates if the waypoints exists. only valid for the last obs
                "own_z_deviation": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64) #this is the deviation of the desired altitude
            }
        )
        
        # first value is heading change, second value is altitude change
        
        # Example: heading continuous, altitude discrete (5 options)
        self.action_space = spaces.Tuple((
            spaces.Box(-1, 1, shape=(1,), dtype=np.float32),   # heading
            spaces.Discrete(2*ALTITUDE_STEPS+1)                                 # altitude: 0,1,2,3,4
        ))
        # initialize bluesky as non-networked simulation node
        if bs.sim is None:
            bs.init(mode='sim', detached=True,workdir=workdir)

        # initialize dummy screen and set correct sim speed
        bs.scr = ScreenDummy()
        bs.stack.stack('DT 1;FF')
        if DEBUG:
        # Only consider nodes with 'pos' attribute for min/max calculations
            latlon_nodes = [(data['lat'], data['lon']) for _, data in self.graph.nodes(data=True) if 'lat' in data and 'lon' in data]
            if latlon_nodes:
                lats = [lat for lat, _ in latlon_nodes]
                lons = [lon for _, lon in latlon_nodes]
                self.max_lat = max(lats)
                self.min_lat = min(lats)
                self.max_lon = max(lons)
                self.min_lon = min(lons)
                self.median_lat = np.median(lats)
                self.median_lon = np.median(lons)
                print(f"Graph loaded with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
                print(f"Graph lat range: {self.min_lat} - {self.max_lat}")
                print(f"Graph lon range: {self.min_lon} - {self.max_lon}")
                print(f"Graph median lat: {self.median_lat} median lon: {self.median_lon}")
            else:
                print("No nodes with 'lat' and 'lon' attributes found in the graph.")
            

        self.px_per_km = self.window_width/(2*stencil_radius_in_km)
        self.stencil_radius_in_km = stencil_radius_in_km
        self.center_point = {"lat":0, "lon":0}
        self.subgraph = None
        self.boundary_vertices = None
        self.intruder_paths = []
        self.current_passed_waypoint_idx = 0
        self.bisector_lines = []
        self.used_node_ids = set()
        
    # ------------------------------------------------------------------ #
    #  Raster-based obstacle generation (works for any graph topology)      #
    # ------------------------------------------------------------------ #

    # Resolution of the offscreen rasterisation buffer.
    # Higher = more accurate contours but slower.  1000 is a good trade-off.
    _RASTER_RES = 1000

    # Minimum area of a void region in *pixels²* to keep (filters noise).
    _MIN_VOID_PX_AREA = 15

    # Maximum number of vertices kept in each simplified contour polygon.
    _MAX_CONTOUR_VERTS = 20

    # --- pixel ↔ lat/lon helpers ------------------------------------------

    def _latlon_to_px(self, lat, lon, buf_w, buf_h):
        """Project lat/lon → pixel (x, y) on the raster buffer."""
        qdr, dis = bs.tools.geo.kwikqdrdist(
            self.center_point['lat'], self.center_point['lon'], lat, lon)
        # dis is in NM; convert to km then to fraction of stencil radius
        frac = (dis * NM2KM) / self.stencil_radius_in_km
        px_x = buf_w / 2 + np.sin(np.deg2rad(qdr)) * frac * buf_w / 2
        px_y = buf_h / 2 - np.cos(np.deg2rad(qdr)) * frac * buf_h / 2
        return int(round(px_x)), int(round(px_y))

    def _px_to_latlon(self, px_x, px_y, buf_w, buf_h):
        """Reverse-project pixel (x, y) → (lat, lon)."""
        # pixel offset from centre
        dx = (px_x - buf_w / 2) / (buf_w / 2) * self.stencil_radius_in_km
        dy = -(px_y - buf_h / 2) / (buf_h / 2) * self.stencil_radius_in_km
        dist_km = np.sqrt(dx**2 + dy**2)
        bearing = np.rad2deg(np.arctan2(dx, dy))  # atan2(east, north)
        dist_nm = dist_km / NM2KM
        # kwikpos: given reference + bearing + distance → new lat/lon
        lat, lon = bs.tools.geo.kwikpos(
            self.center_point['lat'], self.center_point['lon'],
            bearing, dist_nm)
        return lat, lon

    # --- boundary extraction ------------------------------------------------

    @staticmethod
    def _boundary_to_polygon(region_mask, max_verts=48):
        """Extract boundary pixels of a binary region and return an
        angle-sorted polygon of at most *max_verts* vertices.

        Uses ``binary_erosion`` for reliable boundary detection and
        angular sorting from the centroid so edges never cross.

        Returns a list of ``(idx0, idx1)`` pixel-coordinate tuples.
        """
        eroded = ndimage.binary_erosion(region_mask)
        boundary = region_mask & ~eroded
        coords = np.argwhere(boundary)  # (N, 2)
        if len(coords) < 3:
            return []

        # Angular sort from centroid
        cx, cy = coords.mean(axis=0)
        angles = np.arctan2(coords[:, 1] - cy, coords[:, 0] - cx)
        order = np.argsort(angles)

        # Uniform subsample along the angle-sorted list (preserves shape)
        if len(order) > max_verts:
            pick = np.linspace(0, len(order) - 1, max_verts, dtype=int)
            order = order[pick]

        return [(int(coords[i][0]), int(coords[i][1])) for i in order]

    # --- main entry point  -------------------------------------------------

    def _generate_face_obstacles(self):
        """Rasterize corridors onto an offscreen buffer, flood-fill from
        the border to find exterior space, then extract interior void
        regions as obstacle polygons in lat/lon coordinates.

        Each obstacle dict contains:
          - ``coords``   – [(lat, lon), …] polygon of the void region
          - ``path``     – matplotlib.path.Path for point-in-polygon tests
          - ``centroid`` – (lat, lon) of the centroid
        """
        if self.subgraph is None or self.subgraph.number_of_edges() == 0:
            return []

        # Ensure pygame is initialised (needed for offscreen Surface)
        if not pygame.get_init():
            pygame.init()

        W = H = self._RASTER_RES
        # corridor half-width in pixels
        corridor_half_px = max(1, int(
            (AIRWAY_WIDTH / 2.0 * NM2KM) /
            self.stencil_radius_in_km * (W / 2)))
        corridor_px = corridor_half_px * 2 + 1  # full diameter (odd)

        # ---- 1. Rasterize corridors onto a binary mask --------------------
        # We use an offscreen pygame Surface so we can reuse the same
        # thick-line drawing already proven in _render_frame.
        buf = pygame.Surface((W, H))
        buf.fill((0, 0, 0))  # black = empty

        for u, v in self.subgraph.edges():
            pu = self.subgraph.nodes[u]
            pv = self.subgraph.nodes[v]
            if 'lat' not in pu or 'lat' not in pv:
                continue
            x1, y1 = self._latlon_to_px(pu['lat'], pu['lon'], W, H)
            x2, y2 = self._latlon_to_px(pv['lat'], pv['lon'], W, H)
            pygame.draw.line(buf, (255, 255, 255), (x1, y1), (x2, y2),
                             corridor_px)
            # round caps at each node
            pygame.draw.circle(buf, (255, 255, 255), (x1, y1), corridor_half_px)
            pygame.draw.circle(buf, (255, 255, 255), (x2, y2), corridor_half_px)

        # Convert to numpy mask: 1 = corridor, 0 = empty
        arr = pygame.surfarray.pixels3d(buf)          # shape (W, H, 3)
        mask = (arr[:, :, 0] > 128).astype(np.uint8)  # shape (W, H)
        del arr  # release surface lock

        # ---- 2. Flood-fill exterior from border ---------------------------
        # empty_mask: 1 where there is NO corridor
        empty_mask = 1 - mask

        # Label connected components of the empty space
        labeled, num_features = ndimage.label(empty_mask)

        # Find which labels touch the image border → those are exterior
        border_labels = set()
        border_labels.update(labeled[0, :].tolist())   # top row
        border_labels.update(labeled[-1, :].tolist())   # bottom row
        border_labels.update(labeled[:, 0].tolist())    # left col
        border_labels.update(labeled[:, -1].tolist())   # right col
        border_labels.discard(0)  # 0 = corridor, not a void region

        # ---- 3. Extract interior void regions -----------------------------
        interior_labels = set()
        obstacles = []
        for lbl in range(1, num_features + 1):
            if lbl in border_labels:
                continue  # skip exterior

            region_size = np.sum(labeled == lbl)
            if region_size < self._MIN_VOID_PX_AREA:
                continue  # too small, just noise

            interior_labels.add(lbl)

            # Build polygon from boundary pixels
            region_mask = (labeled == lbl)
            poly_px = self._boundary_to_polygon(region_mask,
                                                self._MAX_CONTOUR_VERTS)
            if len(poly_px) < 3:
                continue

            # surfarray pixels3d shape is (W, H, 3) → axis-0 = x, axis-1 = y
            coords_ll = []
            for px_x, px_y in poly_px:
                lat, lon = self._px_to_latlon(px_x, px_y, W, H)
                coords_ll.append((lat, lon))

            if len(coords_ll) < 3:
                continue

            centroid_lat = np.mean([ll[0] for ll in coords_ll])
            centroid_lon = np.mean([ll[1] for ll in coords_ll])

            obstacles.append({
                'coords':   coords_ll,
                'path':     matplotlib.path.Path(coords_ll),
                'centroid': (centroid_lat, centroid_lon),
            })

        # Store raster data for pixel-precise collision detection
        self._obstacle_labeled = labeled
        self._interior_labels = interior_labels

        if DEBUG:
            print(f"Obstacle generation (raster): {num_features} void "
                  f"components found, {len(border_labels)} exterior, "
                  f"{len(obstacles)} interior obstacle(s) kept.")

        return obstacles

    def _get_obs(self):
        ac_idx = bs.traf.id2idx('KL001')

        self.intruder_distance = []
        self.cos_bearing = []
        self.sin_bearing = []
        self.x_difference_speed = []
        self.y_difference_speed = []

        self.waypoint_distance = []
        self.wpt_qdr = []
        self.cos_drift = []
        self.sin_drift = []
    
        self.ac_hdg = bs.traf.hdg[ac_idx]
        self.ac_alt = bs.traf.alt[ac_idx]
        self.own_z_deviation = (self.ac_alt - FLIGHT_LEVEL_M)/2*VERTICAL_SEPARATION_IN_M #z-deviation normalized by vertical separaiton
        self.z_deviation = []
        self.relative_intruder_z_deviation = []
        for i in range(NUM_INTRUDERS):
            int_id = f'INT{i:03d}'
            int_idx = bs.traf.id2idx(int_id)
            if int_idx < 0:
                # Intruder doesn't exist, fill with default values
                self.intruder_distance.append(SENSOR_RANGE)
                self.cos_bearing.append(1)  # cos(0) = 1
                self.sin_bearing.append(0)  # sin(0) = 0
                self.x_difference_speed.append(0)
                self.y_difference_speed.append(0)
                self.z_deviation.append(0)
                continue
            z_dev = self.ac_alt - bs.traf.alt[int_idx]
            self.z_deviation.append(z_dev)
            
            int_qdr, int_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], bs.traf.lat[int_idx], bs.traf.lon[int_idx])
        
            self.intruder_distance.append(int_dis*NM2KM)

            bearing = self.ac_hdg - int_qdr
            bearing = fn.bound_angle_positive_negative_180(bearing)

            self.cos_bearing.append(np.cos(np.deg2rad(bearing)))
            self.sin_bearing.append(np.sin(np.deg2rad(bearing)))

            heading_difference = bs.traf.hdg[ac_idx] - bs.traf.hdg[int_idx]
            x_dif = - np.cos(np.deg2rad(heading_difference)) * bs.traf.gs[int_idx]
            y_dif = bs.traf.gs[ac_idx] - np.sin(np.deg2rad(heading_difference)) * bs.traf.gs[int_idx]

            self.x_difference_speed.append(x_dif)
            self.y_difference_speed.append(y_dif)
            
        # set the waypoints in the observation frame
        self.waypoint_distance = []
        self.cos_wp_bearing = []
        self.sin_wp_bearing = []
        self.waypoint_mask = []
        
        
        for i in range(3):
            if self.current_passed_waypoint_idx + i < len(self.agent_nav_path):
                wp = self.agent_nav_path[self.current_passed_waypoint_idx + i]
                wpt_qdr, wpt_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], wp['lat'], wp['lon'])
                self.waypoint_distance.append(np.clip((wpt_dis * NM2KM)/SENSOR_RANGE,0,1))
        
                self.cos_wp_bearing.append(np.cos(np.deg2rad(wpt_qdr - self.ac_hdg)))
                self.sin_wp_bearing.append(np.sin(np.deg2rad(wpt_qdr - self.ac_hdg)))
                self.waypoint_mask.append(1)
            else:
                self.waypoint_distance.append(0)
                self.cos_wp_bearing.append(0)
                self.sin_wp_bearing.append(0)
                self.waypoint_mask.append(0)
                
        
        observation = {
                "intruder_distance": np.clip(np.array(self.intruder_distance)/SENSOR_RANGE,0,1),
                "cos_difference_pos": np.array(self.cos_bearing),
                "sin_difference_pos": np.array(self.sin_bearing),
                "x_difference_speed": np.array(self.x_difference_speed)/AC_SPD,
                "y_difference_speed": np.array(self.y_difference_speed)/AC_SPD,
                "z_difference_pos": np.clip(np.array(self.z_deviation)/VERTICAL_SEPARATION_IN_M,-1,1),
                "waypoint_distance": np.clip(np.array(self.waypoint_distance)/SENSOR_RANGE,0,1),
                "waypoint_cos_pos": np.array(self.cos_wp_bearing),
                "waypoint_sin_pos": np.array(self.sin_wp_bearing),
                "waypoint_mask": np.array(self.waypoint_mask),
                "own_z_deviation": np.array([self.own_z_deviation])
            }
        
        return observation
    
    
    def _get_cross_track_error(self):
        """
        Helper to calculate the perpendicular distance (NM) from the current airway centerline.
        """
        try:
            ac_idx = bs.traf.id2idx('KL001')
            ac_lat = bs.traf.lat[ac_idx]
            ac_lon = bs.traf.lon[ac_idx]
        except:
            return 0.0

        # Determine the start and end waypoints of the current leg
        # If we are at the last waypoint, look back at the previous segment
        idx = self.current_passed_waypoint_idx
        if idx >= len(self.agent_nav_path) - 1:
            idx = len(self.agent_nav_path) - 2
        
        # Safety check if path is too short
        if idx < 0: 
            return 0.0

        p1 = self.agent_nav_path[idx]
        p2 = self.agent_nav_path[idx+1]

        # 1. Bearing of the path (WayPoint 1 to WayPoint 2)
        qdr_path, _ = bs.tools.geo.kwikqdrdist(p1['lat'], p1['lon'], p2['lat'], p2['lon'])

        # 2. Bearing and distance from WayPoint 1 to Aircraft
        qdr_ac, dist_ac = bs.tools.geo.kwikqdrdist(p1['lat'], p1['lon'], ac_lat, ac_lon)

        # 3. Angle difference (Track Error)
        # We use sin() to find the perpendicular component (Cross Track Distance)
        angle_diff = np.deg2rad(qdr_ac - qdr_path)
        
        # Cross track error in NM (kwikqdrdist returns NM)
        xte = dist_ac * np.sin(angle_diff)
        
        return xte

    def _get_corridor_penalty(self):
        # calculate the penalty if the agent leaves the corridor sparse penalty
        xte = self._get_cross_track_error()
        
        # AIRWAY_WIDTH is the total width (e.g. 8 NM), so deviation limit is half that
        half_width = AIRWAY_WIDTH / 2.0
        
        if abs(xte) > half_width:
            return CORRIDOR_LEAVE_PENALTY
        
        return 0.0
    
    def _get_drift_penalty(self):
        # drift penalty between the current waypoint and the previous one, stay close to the centerline of the airway continuous
        xte = self._get_cross_track_error()
        half_width = AIRWAY_WIDTH / 2.0
        
        # Normalize the deviation: 0.0 at center, 1.0 at the edge of the airway
        # We clip at 1.0 to ensure the penalty doesn't explode if they go way off track 
        # (the corridor penalty handles the "way off track" discrete case)
        normalized_deviation = np.clip(abs(xte) / half_width, 0, 1)
        
        # Apply penalty scaled by the factor
        # If DRIFT_PENALTY is -0.5, then max penalty is -0.5 at the edge, 0 at center.
        return normalized_deviation * DRIFT_PENALTY
    
    def _get_intrusion_penalty(self):
        # intrusion penalty if the intruder is within a certain distance of the agent, sparse penalty
        # can have either have a horizontal seperation or vertical seperation, or a combination of both
        # if neither horizontal nor vertical seperation is respected apply the penalty.
        
        total_penalty = 0.0
        
        try:
            own_idx = bs.traf.id2idx('KL001')
            own_lat = bs.traf.lat[own_idx]
            own_lon = bs.traf.lon[own_idx]
            own_alt = bs.traf.alt[own_idx]
        except:
            # If agent doesn't exist, no penalty calculation needed (or max penalty elsewhere)
            return 0.0

        for i in range(NUM_INTRUDERS):
            int_id = f'INT{i:03d}'
            int_idx = bs.traf.id2idx(int_id)
            
            # Skip if intruder doesn't exist
            if int_idx < 0:
                continue
                
            int_lat = bs.traf.lat[int_idx]
            int_lon = bs.traf.lon[int_idx]
            int_alt = bs.traf.alt[int_idx]

            # Horizontal Distance (NM)
            _, dist_nm = bs.tools.geo.kwikqdrdist(own_lat, own_lon, int_lat, int_lon)
            
            # Vertical Distance (Meters)
            dist_vert_m = abs(own_alt - int_alt)

            # Check for Loss of Separation (LOS)
            # LOS occurs if BOTH horizontal AND vertical constraints are violated simultaneously
            horizontal_violation = dist_nm < INTRUSION_DISTANCE
            vertical_violation = dist_vert_m < VERTICAL_SEPARATION_IN_M
            terminated = False
            if horizontal_violation and vertical_violation and dist_nm < CRASH_DISTANCE_HORIZONTAL and dist_vert_m < CRASH_DISTANCE_VERTICAL:
                total_penalty += CRASH_PENALTY
                terminated = True
            elif horizontal_violation and vertical_violation:
                total_penalty += INTRUSION_PENALTY
            
                
        return total_penalty,terminated
    
    def _altitude_penalty(self):
        # penalize the agent for deviating fromt the target penalty,
        # continuous from the discrete FL steps normalized to 0 and 1
        ac_alt = bs.traf.alt[bs.traf.id2idx('KL001')]
        alt_diff = abs(ac_alt - FLIGHT_LEVEL_M)
        max_diff = ALTITUDE_STEPS * D_ALTITUDE
        normalized_diff = np.clip(alt_diff / max_diff, 0, 1)
        return normalized_diff * ALTITUDE_PENALTY

    def _get_obstacle_penalty(self):
        """Check if the agent is inside a void-obstacle region using the
        raster mask.  Returns ``OBSTACLE_PENALTY`` on collision, else 0."""
        if not hasattr(self, '_obstacle_labeled') or not self._interior_labels:
            return 0.0
        try:
            ac_idx = bs.traf.id2idx('KL001')
            ac_lat = bs.traf.lat[ac_idx]
            ac_lon = bs.traf.lon[ac_idx]
        except Exception:
            return 0.0

        W = H = self._RASTER_RES
        px_x, px_y = self._latlon_to_px(ac_lat, ac_lon, W, H)
        if 0 <= px_x < W and 0 <= px_y < H:
            label = self._obstacle_labeled[px_x, px_y]
            if label in self._interior_labels:
                return OBSTACLE_PENALTY
        return 0.0

    def _get_reward(self):
        
        terminated = False
        waypoints_passed = self._check_pass_waypoint_bisector_line()
        drift_penalty = self._get_drift_penalty()
        corridor_penalty = self._get_corridor_penalty()
        intrusion_penalty,terminated = self._get_intrusion_penalty()
        altitude_penalty = self._altitude_penalty()
        obstacle_penalty = self._get_obstacle_penalty()
        reach_reward = waypoints_passed * REACH_REWARD
        
        
        if self.current_passed_waypoint_idx == len(self.agent_nav_path)-1:
            # give a big reward for reaching the final waypoint
            reach_reward += REACH_REWARD * 5
            terminated = True
        
        reward = reach_reward + drift_penalty + corridor_penalty + intrusion_penalty + altitude_penalty + obstacle_penalty
        return reward, terminated
        
    
    def _calculate_all_bisector_lines(self):
        """Pre-calculate bisector lines for all intermediate waypoints in the agent nav path."""
        self.bisector_lines = []
        if not self.agent_nav_path or len(self.agent_nav_path) < 3:
            return

        # Bisector lines exist for waypoints 1 through len-2 (all that have both a predecessor and successor)
        for i in range(1, len(self.agent_nav_path) - 1):
            wp_prev = self.agent_nav_path[i - 1]
            wp_target = self.agent_nav_path[i]
            wp_next = self.agent_nav_path[i + 1]

            # Calculate inbound and outbound tracks
            qdr_in, _ = bs.tools.geo.kwikqdrdist(wp_prev['lat'], wp_prev['lon'], wp_target['lat'], wp_target['lon'])
            qdr_out, _ = bs.tools.geo.kwikqdrdist(wp_target['lat'], wp_target['lon'], wp_next['lat'], wp_next['lon'])

            # Calculate average angle properly handling 360 wrap
            sum_vectors = np.exp(1j * np.deg2rad(qdr_in)) + np.exp(1j * np.deg2rad(qdr_out))
            avg_angle_rad = np.angle(sum_vectors)
            bisector_qdr = np.rad2deg(avg_angle_rad)

            self.bisector_lines.append({
                "lat": wp_target['lat'],
                "lon": wp_target['lon'],
                "normal_qdr": bisector_qdr,
                "waypoint_idx": i  # index in agent_nav_path
            })

    def _check_pass_waypoint_bisector_line(self):
        ac_idx = bs.traf.id2idx('KL001')
        ac_lat = bs.traf.lat[ac_idx]
        ac_lon = bs.traf.lon[ac_idx]

        # The bisector line index corresponds to current_passed_waypoint_idx
        # (bisector_lines[0] is for waypoint index 1, bisector_lines[k] is for waypoint index k+1)
        bisector_idx = self.current_passed_waypoint_idx  # maps to the next waypoint to pass
        waypoints_passed = 0
        passed_this_step = False
        while True:
            if bisector_idx < len(self.bisector_lines):
                bl = self.bisector_lines[bisector_idx]

                # Check aircraft position relative to the bisector line
                qdr_wp_ac, dist_wp_ac = bs.tools.geo.kwikqdrdist(bl['lat'], bl['lon'], ac_lat, ac_lon)
                angle_diff = fn.bound_angle_positive_negative_180(qdr_wp_ac - bl['normal_qdr'])

                if abs(angle_diff) < 90 and (dist_wp_ac * NM2KM < 2*AIRWAY_WIDTH): #This means the aircraft can pass the waypoint 1 corridor width to each side of the bisector line
                    self.current_passed_waypoint_idx += 1
                    passed_this_step = True
                    #print(f"Passed waypoint {self.current_passed_waypoint_idx}")

            elif self.current_passed_waypoint_idx + 1 < len(self.agent_nav_path):
                # Final waypoint logic (no bisector available)
                wp_target = self.agent_nav_path[self.current_passed_waypoint_idx + 1]
                _, dist = bs.tools.geo.kwikqdrdist(ac_lat, ac_lon, wp_target['lat'], wp_target['lon'])

                if (dist * NM2KM) < WAYPOINT_REACH_DISTANCE:
                    self.current_passed_waypoint_idx += 1
                    passed_this_step = True
                    #print(f"Reached final waypoint {self.current_passed_waypoint_idx}")

            if not passed_this_step:
                break
            else:
                waypoints_passed += 1
                passed_this_step = False
                bisector_idx = self.current_passed_waypoint_idx
        
        return waypoints_passed
        
    def _get_action(self, action):
        action_hdg = self.ac_hdg + action[0] * D_HEADING

        bs.stack.stack(f"HDG KL001 {action_hdg}")
        
        
        shifted_action = action[1] - ALTITUDE_STEPS # shift from 0,.., n to -1/2n,..., 0,..., 1/2n
        target_altitude = FLIGHT_LEVEL_M + shifted_action * D_ALTITUDE
        
        ac_idx = bs.traf.id2idx('KL001')
        
        current_selalt = bs.traf.selalt[ac_idx]
        current_altitude = bs.traf.alt[ac_idx]

        # Only allow new SELALT if close to previous SELALT (within half a step)
        if abs(current_altitude - current_selalt) < ALT_REACH_DISTANCE:
            bs.traf.selalt[ac_idx] = target_altitude
            if target_altitude > current_altitude:
                bs.traf.selvs[ac_idx] = CLIMB_RATE
            elif current_altitude > target_altitude:
                bs.traf.selvs[ac_idx] = DECENT_RATE
    
    
    def step(self, action):
        
        self._get_action(action)

        action_frequency = ACTION_FREQUENCY
        for i in range(action_frequency):
            bs.sim.step()
            if self.render_mode == "human":
                observation =  self._get_obs()
                self._render_frame()

        observation = self._get_obs()
        reward, terminated = self._get_reward()

        info =  {} #self._get_info()

        # bluesky reset?? bs.sim.reset()
        if terminated:
            for acid in bs.traf.id:
                idx = bs.traf.id2idx(acid)
                bs.traf.delete(idx)

        return observation, reward, terminated, False, info
    
    def _reset_class_variables(self):
        self.center_point = {"lat":0, "lon":0}
        self.subgraph = None
        self.boundary_vertices = None
        self.agent_nav_path = None
        self.intruder_paths = []
        self.current_passed_waypoint_idx = 0
        self.bisector_lines = []
        self.used_node_ids = set()
        self.face_obstacles = []
        self._obstacle_labeled = None
        self._interior_labels = set()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_class_variables()
        bs.traf.reset()
        number_of_vertices = 0
        
        while number_of_vertices < 10:
            random_node = self.np_random.choice(list(self.graph.nodes))
            self.center_point = {"lat": self.graph.nodes[random_node]['lat'], "lon": self.graph.nodes[random_node]['lon']}
            subgraph = self._get_subgraph_around_waypoint((self.center_point["lat"], self.center_point["lon"]), self.stencil_radius_in_km)
            self.subgraph = subgraph
            number_of_vertices = self.subgraph.number_of_nodes()
        
        self.boundary_vertices = self._get_boundary_vertices()
        self.face_obstacles = self._generate_face_obstacles()
        
        self.agent_nav_path = self._get_agent_nav_path(self.boundary_vertices)
        self._calculate_all_bisector_lines()
        self.intruder_paths = []
        for _ in range(NUM_INTRUDERS-1):
            intruder_path = self._intersecting_nav_path(self.agent_nav_path,self.boundary_vertices)
            
            if len(intruder_path) > MIN_ROUTE_LENGTH:
                # randomly drop the first 1-3 waypoints on each end
                drop_start = self.np_random.integers(0, 3)
                
                intruder_path = intruder_path[drop_start:]
                    
                if intruder_path:
                    self.intruder_paths.append(intruder_path)
        # add a last intruder path where its just the reverse of the agent path
        conflict_intruder_path = self.agent_nav_path[::-1]
        self.intruder_paths.append(conflict_intruder_path)
        # Collect all node IDs used in any path for filtered rendering
        self.used_node_ids = set()
        for wp in self.agent_nav_path:
            self.used_node_ids.add(wp['id'])
        for intruder_path in self.intruder_paths:
            for wp in intruder_path:
                self.used_node_ids.add(wp['id'])
        
        self._spawn_agent()
        self._spawn_intruders()
        if DEBUG:
            for i in self.intruder_paths:
                print(f"Intruder path with {len(i)} waypoints")
        self._render_frame()
        
        return self._get_obs(), {} #self._get_info()
    
    def _spawn_agent(self):
        # calculate heading between first two waypoints
        # Now agent_nav_path contains dicts, so we access lat/lon directly
        p0 = self.agent_nav_path[0]
        p1 = self.agent_nav_path[1]
        bearing, _ = bs.tools.geo.kwikqdrdist(p0['lat'], p0['lon'], p1['lat'], p1['lon'])
        # spawn the agent on the first waypoint with the calculated heading and speed of AC_SPD
        bs.traf.cre('KL001',actype="A320",acspd=AC_SPD,acalt=FLIGHT_LEVEL_M,aclat = p0["lat"],aclon=p0["lon"],achdg=bearing)
        
    def _spawn_intruders(self):
        for i, intruder_path in enumerate(self.intruder_paths):
            acid = f'INT{i:03d}'
            idx = bs.traf.id2idx(f'INT{i:03d}')
            if intruder_path:
                p0 = intruder_path[0]
                p1 = intruder_path[1]
                bearing, _ = bs.tools.geo.kwikqdrdist(p0['lat'], p0['lon'], p1['lat'], p1['lon'])
                
                ac_alt = FLIGHT_LEVEL_M + self.np_random.integers(-INTRUDER_ALT_SPANRANGE_IN_1000FT*VERTICAL_SEPARATION_IN_M, INTRUDER_ALT_SPANRANGE_IN_1000FT*VERTICAL_SEPARATION_IN_M)
                bs.traf.cre(acid,actype="A320",acspd=AC_SPD,acalt=ac_alt,aclat = p0["lat"],aclon=p0["lon"],achdg=bearing)

                route_obj = bs.traf.ap.route[idx] # Get the specific route instance

            # 3. Add Waypoints
            for j, node in enumerate(intruder_path[1:]):
                
                # Name: Must be unique-ish (e.g., INT001_WP1)
                wp_name = f"{acid}_WP{j}"
                
                # Type: Access the constant 'wplatlon' directly from the object
                # This ensures we match 'if wptype == Route.wplatlon:' in the source code
                wp_type = route_obj.wplatlon 
                
                route_obj.addwpt(
                    idx,           # Arg 1: iac (Aircraft Index) - REQUIRED
                    wp_name,       # Arg 2: name (String) - REQUIRED
                    wp_type,       # Arg 3: wptype (Int Constant) - REQUIRED
                    node['lat'],   # Arg 4: lat
                    node['lon'],   # Arg 5: lon
                    FLIGHT_LEVEL_M,  # Arg 6: alt (Optional but recommended)
                    AC_SPD         # Arg 7: spd (Optional but recommended)
                )
                bs.traf.swlnav[idx] = 1
                bs.traf.actwp.turnbank[idx]= 45.0
    
    def _get_subgraph_around_waypoint(self, waypoint, stencil):
        # get the nodes within the stencil radius
        nodes_in_stencil = []
        for node, data in self.graph.nodes(data=True):
            if 'lat' in data and 'lon' in data:
                # waypoint should be (lat, lon)
                _, dist = bs.tools.geo.kwikqdrdist(data['lat'], data['lon'], waypoint[0], waypoint[1])
                if dist * NM2KM <= stencil:
                    nodes_in_stencil.append(node)
        
        # create the subgraph
        subgraph = self.graph.subgraph(nodes_in_stencil).copy()
        # Remove nodes with no edges
        subgraph.remove_nodes_from(list(nx.isolates(subgraph)))
        
        return subgraph
    
    def _get_agent_nav_path(self, boundary_vertices):
        # select a random start and end point from the boundary vertices
        if len(boundary_vertices) < 2:
            return []
        start, end = self.np_random.choice(boundary_vertices, size=2, replace=False)
        # find the shortest path between them
        # use networkx shortest path algorithm with weight as distance
        try:
            path_ids = nx.shortest_path(self.subgraph, source=start, target=end, weight='weight')
            # Convert list of IDs to list of node data dicts (with 'id' added)
            path_data = []
            for node_id in path_ids:
                node_data = self.subgraph.nodes[node_id].copy()
                node_data['id'] = node_id
                path_data.append(node_data)
            return path_data
        except nx.NetworkXNoPath:
            return []
        
    def _intersecting_nav_path(self, agent_path, boundary_vertices):
        # take two random boundary vertices that are not in the agent path
        if len(boundary_vertices) < 4:
            return []
            
        # extract just IDs for comparison since agent_path is now enriched
        agent_path_ids = [node['id'] for node in agent_path]
        
        # the boundary vertices in the agent path are 0 and -1, so we need to exclude them from the random selection
        # so just remove them from the list of boundary vertices and then select from the remaining ones
        available_vertices = [v for v in boundary_vertices if v not in agent_path_ids]
        if len(available_vertices) < 2:
            return []
        
        start, end = self.np_random.choice(available_vertices, size=2, replace=False)
        # take a random point on the agent path as the intersection point
        if len(agent_path) < 3:
            return []
            
        # Select intersection node from the enriched path
        intersection_node_data = self.np_random.choice(agent_path[1:-1])
        intersection_point = intersection_node_data['id']
        
        # find the shortest path from start to intersection point and from end to intersection point
        try:
            path1 = nx.shortest_path(self.subgraph, source=start, target=intersection_point, weight='weight')
            path2 = nx.shortest_path(self.subgraph, source=end, target=intersection_point, weight='weight')
            # combine the two paths to create the intersecting path
            # path1 goes start->intersection, path2 goes end->intersection
            # we want start->intersection->end? The original code did: path1[:-1] + path2[::-1]
            intersecting_path_ids = path1[:-1] + path2[::-1]
            
            # Convert list of IDs to list of node data dicts
            path_data = []
            for node_id in intersecting_path_ids:
                node_data = self.subgraph.nodes[node_id].copy()
                node_data['id'] = node_id
                path_data.append(node_data)
                
            return path_data
        except nx.NetworkXNoPath:
            return []
    
    def _get_boundary_vertices(self):
        # We need at least a basic structure to trace
        if self.subgraph is None or self.subgraph.number_of_nodes() < 3:
            return list(self.subgraph.nodes()) if self.subgraph else []

        # 1. Find the starting node (minimum latitude)
        # By picking the absolute lowest point, we guarantee we start on the outer face.
        start_node = min(self.subgraph.nodes(), key=lambda n: self.subgraph.nodes[n]['lat'])
        
        def get_angle(n1, n2):
            """Calculate the polar angle between two nodes."""
            lat1, lon1 = self.subgraph.nodes[n1]['lat'], self.subgraph.nodes[n1]['lon']
            lat2, lon2 = self.subgraph.nodes[n2]['lat'], self.subgraph.nodes[n2]['lon']
            return math.atan2(lat2 - lat1, lon2 - lon1)

        # 2. Pick the initial outgoing edge
        neighbors = list(self.subgraph.neighbors(start_node))
        if not neighbors:
            return [start_node]
            
        # Sort the starting neighbors to find the rightmost path to begin the walk
        neighbors.sort(key=lambda n: get_angle(start_node, n))
        first_neighbor = neighbors[0] 
        
        boundary_vertices = [start_node]
        
        # Setup the walking pointers
        prev_node = start_node
        curr_node = first_neighbor
        
        # Safety limit to prevent infinite loops if the graph data has structural errors
        max_iters = self.subgraph.number_of_edges() * 3
        
        for _ in range(max_iters):
            # If we returned to the exact starting point, the outer perimeter is closed
            if curr_node == start_node:
                break
                
            boundary_vertices.append(curr_node)
            curr_neighbors = list(self.subgraph.neighbors(curr_node))
            
            # If it is a dead end, we just turn around and walk back
            if len(curr_neighbors) == 1:
                next_node = curr_neighbors[0]
            else:
                # Sort all available outgoing paths by their angle
                curr_neighbors.sort(key=lambda n: get_angle(curr_node, n))
                
                # Find the path we just came from in that sorted list
                incoming_idx = curr_neighbors.index(prev_node)
                
                # The right-hand rule means taking the very next path in the rotation
                next_idx = (incoming_idx + 1) % len(curr_neighbors)
                next_node = curr_neighbors[next_idx]
                
            # Move forward one step
            prev_node = curr_node
            curr_node = next_node
            
        # Clean up duplicates in case the algorithm walked in and out of a dead-end branch
        return list(dict.fromkeys(boundary_vertices))
    
    
    def _pre_render(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
            
            
        #process pygame events to prevent "not responding" window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                exit()
                return
            
    def lat_lon_to_screen_coordinates (self,lat,lon,*args,**kwargs)->tuple:
        qdr, dis = bs.tools.geo.kwikqdrdist(self.center_point["lat"],self.center_point["lon"], lat, lon)
        
        x_pos = (self.window_width/2)+(np.sin(np.deg2rad(qdr))*(dis * NM2KM)*self.px_per_km)
        y_pos = (self.window_height/2)-(np.cos(np.deg2rad(qdr))*(dis * NM2KM)*self.px_per_km)
        if self.show_altitude_in_rendering:
            y_pos+=100
        return x_pos,y_pos
        
            
    def _post_render(self,canvas):
        self.window.blit(canvas, canvas.get_rect())
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])
        
    def _plot_scale(self,canvas):
        # plot a scale of 10 km in the bottom left corner
        scale_length_in_km = 10
        scale_length_in_px = scale_length_in_km * self.px_per_km
        pygame.draw.line(canvas, COLORS["SCALE_LINE"], (50, self.window_height-50), (50+scale_length_in_px, self.window_height-50), int(0.5*self.px_per_km))
        font = pygame.font.SysFont(None, 24)
        text = font.render(f"{scale_length_in_km} km", True, COLORS["SCALE_LINE"])
        canvas.blit(text, (50, self.window_height-80))
        return canvas
    
    def _draw_vercial_seperation(self,canvas,stepsize_in_feet = 1000,steps_from_fl=2):
        # draw a horizontal line in the middle of y = 100 
        # thats where FL340 
        
        # Total vertical range covered by the visualization in pixels
        viz_height_px = 200
        
        # first draw a rectangle  with solid color over the 200 pixel height to make it look better
        pygame.draw.rect(canvas, COLORS["VERTICAL_BG"], pygame.Rect(0, 0, self.window_width, viz_height_px))
        
        center_y = viz_height_px // 2
        # Pixels per step (one stepsize_in_feet interval)
        px_per_step = viz_height_px / (2 * steps_from_fl)
        # Pixels per foot in this specific visualization
        px_per_foot = px_per_step / stepsize_in_feet
        
        labels = []
        for i in range(-steps_from_fl,steps_from_fl+1):
            # FL is at center_y, higher FLs go up (smaller y), lower FLs go down
            y = int(center_y - i * px_per_step)
            labels.append(f"FL{(FLIGHT_LEVEL*100 + (i*stepsize_in_feet))//100}")
            
            
            color = COLORS["GRID_LINES"]
            
            pygame.draw.line(canvas, color, (0, y), (self.window_width, y), 1)
            font = pygame.font.SysFont(None, 24)
            
            
            text = font.render(labels[-1], True, COLORS["TEXT"])
            text_height = text.get_height()
            
            # Position text based on position relative to center
            if i == steps_from_fl:
                # Above center (higher FL): text below the line
                text_y = y + 5
            elif i == -steps_from_fl:
                # Below center (lower FL): text above the line
                text_y = y - text_height - 5
            else:
                # Center line: text vertically centered
                text_y = y - text_height // 2
            
            canvas.blit(text, (10, text_y))
            
        # draw intruders in the visualization 
        own_idx = bs.traf.id2idx('KL001')
        for i in range(NUM_INTRUDERS):
            try:
                int_idx = bs.traf.id2idx(f'INT{i:03d}')
                if int_idx < 0:
                    continue
                
                # 1. Calculate Y position based on foot-difference from your reference FL
                int_alt_ft = bs.traf.alt[int_idx] / FT_TO_M
                alt_diff_ft = int_alt_ft - (FLIGHT_LEVEL * 100)
                
                # Vertical Position: Center (100) minus the displacement
                y_pos = int(center_y - (alt_diff_ft * px_per_foot))
                
                int_lat = bs.traf.lat[int_idx]
                int_lon = bs.traf.lon[int_idx]
                
                x_pos, _ = self.lat_lon_to_screen_coordinates(int_lat, int_lon)
                
                
                
                # 2. Calculate Box Dimensions
                # Width uses horizontal scaling (KM/NM to Pixels)
                intrusion_width_in_px = int((INTRUSION_DISTANCE * NM2KM) * self.px_per_km*2)
                
                # Height uses vertical scaling (1000 feet converted to pixels)
                # This ensures 1000ft separation always spans exactly one 'step' in your grid
                intrusion_height_in_px = int(2000 * px_per_foot)
                
                
                # Check for conflict with ownship if ownship exists
                color = COLORS["INTRUDER_SAFE"] # Default grey
                pygame.draw.circle(canvas, color, (x_pos, y_pos), 5)
                if own_idx >= 0:
                    own_lat = bs.traf.lat[own_idx]
                    own_lon = bs.traf.lon[own_idx]
                    own_alt = bs.traf.alt[own_idx]
                    _, int_dis = bs.tools.geo.kwikqdrdist(own_lat, own_lon, int_lat, int_lon)
                    int_alt = bs.traf.alt[int_idx]
                    
                    if int_dis < INTRUSION_DISTANCE and abs(int_alt - own_alt) < VERTICAL_SEPARATION_IN_M:
                        color = COLORS["INTRUDER_CONFLICT"] # Red if conflict
                
                
                intrusion_rect = pygame.Rect(
                    x_pos - intrusion_width_in_px // 2, 
                    y_pos - intrusion_height_in_px // 2, 
                    intrusion_width_in_px, 
                    intrusion_height_in_px
                )
                
                pygame.draw.rect(canvas, color, intrusion_rect, 1)
            except ValueError:
                continue
            
        # draw ownhsip in the visualization
        own_altitude = bs.traf.alt[own_idx] / FT_TO_M
        alt_diff_ft = own_altitude - (FLIGHT_LEVEL * 100)
        x_pos,_ = self.lat_lon_to_screen_coordinates(bs.traf.lat[own_idx], bs.traf.lon[own_idx])
        own_y_pos = int(center_y - (alt_diff_ft * px_per_foot))
        pygame.draw.circle(canvas, COLORS["OWNSHIP"], (x_pos, own_y_pos), 5)
        
        
        return canvas
        
        
    def _draw_obstacles(self, canvas):
        """Draw the void-obstacle polygons (the free space inside faces
        that is NOT covered by the airway corridor width)."""
        if not hasattr(self, 'face_obstacles'):
            return

        for obs in self.face_obstacles:
            void_screen = []
            for lat, lon in obs['coords']:
                x, y = self.lat_lon_to_screen_coordinates(lat, lon)
                void_screen.append((x, y))

            if len(void_screen) > 2:
                pygame.draw.polygon(canvas, (40, 20, 20), void_screen, 0)   # dark-red fill
                pygame.draw.polygon(canvas, (255, 50, 50), void_screen, 1)  # red outline
    
    def _render_frame(self):
        self._pre_render()
        canvas = pygame.Surface(self.window_size)
        canvas.fill(COLORS["BACKGROUND"])
        
        canvas = self._plot_scale(canvas)
        
        
        agent_path_ids = [p['id'] for p in self.agent_nav_path]
        edge_coords = []
        for u, v in self.subgraph.edges():
            # When plot_all_points is False, only show edges where both nodes are used in a path
            if not self.plot_all_points and (u not in self.used_node_ids or v not in self.used_node_ids):
                continue
            pos_u = self.subgraph.nodes[u]
            pos_v = self.subgraph.nodes[v]
            if 'lat' in pos_u and 'lon' in pos_u and 'lat' in pos_v and 'lon' in pos_v:
                x1, y1 = self.lat_lon_to_screen_coordinates(pos_u['lat'], pos_u['lon'])
                x2, y2 = self.lat_lon_to_screen_coordinates(pos_v['lat'], pos_v['lon'])
                
                # both nodes are in the agent_path add different color
                color = COLORS["AIRWAY"]
                
                if u in agent_path_ids and v in agent_path_ids:
                    color = COLORS["AIRWAY_ACTIVE"]         
                edge_coords.append(((int(x1), int(y1)), (int(x2), int(y2)),color))
        
        # First pass: draw airways
        for start, end,_ in edge_coords:
            pygame.draw.line(canvas, COLORS["AIRWAY_CORRIDOR"], start, end, int(AIRWAY_WIDTH*NM2KM*self.px_per_km))
            #also draw a cirlce at the start and end of each edge to make it look better
            pygame.draw.circle(canvas, COLORS["AIRWAY_CORRIDOR"], start, int(AIRWAY_WIDTH/2*NM2KM*self.px_per_km))
            pygame.draw.circle(canvas, COLORS["AIRWAY_CORRIDOR"], end, int(AIRWAY_WIDTH/2*NM2KM*self.px_per_km))

        self._draw_obstacles(canvas)
        # Draw all bisector lines
        if DEBUG:
            for bi, bl in enumerate(self.bisector_lines):
                wp_lat = bl["lat"]
                wp_lon = bl["lon"]
                normal_qdr = bl["normal_qdr"]
                
                cx, cy = self.lat_lon_to_screen_coordinates(wp_lat, wp_lon)
                
                # The line is perpendicular to the normal_qdr
                line_angle_1 = np.deg2rad(normal_qdr + 90)
                line_angle_2 = np.deg2rad(normal_qdr - 90)
                
                line_len = AIRWAY_WIDTH*NM2KM*self.px_per_km  # Extend the line across the airway corridor
                
                p1_x = cx + np.sin(line_angle_1) * line_len
                p1_y = cy - np.cos(line_angle_1) * line_len # Y is inverted in screen coords
                
                p2_x = cx + np.sin(line_angle_2) * line_len
                p2_y = cy - np.cos(line_angle_2) * line_len
                
                # Highlight the next bisector to pass vs already-passed ones
                if bi < self.current_passed_waypoint_idx:
                    color = COLORS["BISECTOR_PASSED"]  # Dim purple for already-passed
                elif bi == self.current_passed_waypoint_idx:
                    color = COLORS["BISECTOR_ACTIVE"]   # Bright magenta for next to pass
                else:
                    color = COLORS["BISECTOR_FUTURE"] # Light purple for upcoming
                
                pygame.draw.line(canvas, color, (p1_x, p1_y), (p2_x, p2_y), 3)
                
                # Draw the normal vector to show "forward" direction
                n_x = cx + np.sin(np.deg2rad(normal_qdr)) * 30
                n_y = cy - np.cos(np.deg2rad(normal_qdr)) * 30
                pygame.draw.line(canvas, color, (cx, cy), (n_x, n_y), 1)

        # Second pass: draw center lines
        for start, end,color in edge_coords:
            pygame.draw.line(canvas,color, start, end, 1)

        for node in self.subgraph.nodes:
            # When plot_all_points is False, only show nodes used in a path
            if not self.plot_all_points and node not in self.used_node_ids:
                continue
            data = self.subgraph.nodes[node]
            if 'lat' in data and 'lon' in data:
                x, y = self.lat_lon_to_screen_coordinates(data['lat'], data['lon'])
                
                color = COLORS["WAYPOINT"]
                # Check if node id is in the agent path (which is now a list of dicts)
                if self.agent_nav_path[self.current_passed_waypoint_idx+1]["id"] == node:
                    color = COLORS["WAYPOINT_ACTIVE"]
                
                pygame.draw.circle(canvas, color, (int(x), int(y)), 3) 
     
        
        #DRAW INTRUDERS 
        try:
            own_idx = bs.traf.id2idx('KL001')
        except:
            own_idx = -1

        for i in range(NUM_INTRUDERS):
            try:
                int_idx = bs.traf.id2idx(f'INT{i:03d}')
                if int_idx < 0:
                    continue
                
                int_lat = bs.traf.lat[int_idx]
                int_lon = bs.traf.lon[int_idx]
                int_hdg = bs.traf.hdg[int_idx]
                ac_spd = bs.traf.cas[int_idx]
                
                x_pos, y_pos = self.lat_lon_to_screen_coordinates(int_lat, int_lon)
                
                own_alt = bs.traf.alt[own_idx]
                int_alt = bs.traf.alt[int_idx] 
                dif_alt = abs(own_alt - int_alt) 
                
                # Check for conflict with ownship if ownship exists
                color = COLORS["INTRUDER_SAFE"] # Default grey
                if own_idx >= 0:
                    own_lat = bs.traf.lat[own_idx]
                    own_lon = bs.traf.lon[own_idx]
                    _, int_dis = bs.tools.geo.kwikqdrdist(own_lat, own_lon, int_lat, int_lon)
                    if int_dis < INTRUSION_DISTANCE and dif_alt < VERTICAL_SEPARATION_IN_M:
                        color = COLORS["INTRUDER_CONFLICT"] # Red if conflict

                # Draw intruder position
                pygame.draw.circle(canvas, color, (int(x_pos), int(y_pos)), 5)

                # Draw heading vector
                # Length of heading vector in pixels (e.g., representing 2km)
                heading_len_km = 240 * (ac_spd) / 1000  # Scale by speed for better visualization
                heading_len_px = heading_len_km * self.px_per_km
                
                end_x = x_pos + np.sin(np.deg2rad(int_hdg)) * heading_len_px
                end_y = y_pos - np.cos(np.deg2rad(int_hdg)) * heading_len_px
                
                pygame.draw.line(canvas, color, (int(x_pos), int(y_pos)), (int(end_x), int(end_y)), 2)

                # Draw intrusion circle (optional, or maybe just around ownship)
                # If we want to visualize the protection zone around the intruder:
                radius_px = int(INTRUSION_DISTANCE * NM2KM * self.px_per_km)
                pygame.draw.circle(canvas, color, (int(x_pos), int(y_pos)), radius_px, 1)

            except ValueError:
                # Intruder not found
                print(ValueError)
                continue
        
        
        # Draw agent heading vector
        try:
            own_idx = bs.traf.id2idx('KL001')
            if own_idx >= 0:
                own_lat = bs.traf.lat[own_idx]
                own_lon = bs.traf.lon[own_idx]
                own_hdg = bs.traf.hdg[own_idx]
                ac_spd = bs.traf.cas[own_idx]

                x_pos, y_pos = self.lat_lon_to_screen_coordinates(own_lat, own_lon)
                


                
                # Intrusion Circle
                radius_px = int(INTRUSION_DISTANCE * NM2KM * self.px_per_km)
                pygame.draw.circle(canvas, COLORS["OWNSHIP"], (x_pos, y_pos), radius_px, 1)

                # Draw heading vector for the agent
                heading_len_km = 240 * (ac_spd ) / 1000  # Scale by speed for better visualization
                
                
                heading_len_px = heading_len_km * self.px_per_km

                end_x = x_pos + np.sin(np.deg2rad(own_hdg)) * heading_len_px
                end_y = y_pos - np.cos(np.deg2rad(own_hdg)) * heading_len_px
                
                # Draw intruder position
                pygame.draw.circle(canvas, COLORS["OWNSHIP"], (int(x_pos), int(y_pos)), 5)

                pygame.draw.line(canvas, COLORS["OWNSHIP"], (int(x_pos), int(y_pos)), (int(end_x), int(end_y)), 2)
        except ValueError:
            # Agent not found
            print(ValueError)
            
            
        if self.show_altitude_in_rendering:
            canvas = self._draw_vercial_seperation(canvas)

        self._post_render(canvas)
        



if __name__ == "__main__":
    env = NavWaypointEvadeEnv(render_mode="human",window_height=500,window_width=500,stencil_radius_in_km=400,show_altitude_in_rendering=True,plot_all_points=True)
    env.metadata["render_fps"] = 20
    env.reset(seed=48)
    env.reset()
    # # We use env.np_random for consistency with the seeded environment
    # nodes_list = list(env.graph.nodes)
    # if nodes_list:
    #     random_node = env.np_random.choice(nodes_list)
    #     env.center_point = {"lat": env.graph.nodes[random_node]['lat'], "lon": env.graph.nodes[random_node]['lon']}
    #     subgraph = env._get_subgraph_around_waypoint((env.graph.nodes[random_node]['lat'], env.graph.nodes[random_node]['lon']), 100)
    #     env.subgraph = subgraph
    #     #print number of nodes in subgraph
    #     print(f"Subgraph has {subgraph.number_of_nodes()} nodes and {subgraph.number_of_edges()} edges")
    #     env.center_point = {"lat": env.graph.nodes[random_node]['lat'], "lon": env.graph.nodes[random_node]['lon']}
        
    #     boundary_vertices = env._get_boundary_vertices()
        
    #     agent_nav_path = env._get_agent_nav_path(boundary_vertices)
    #     env.agent_nav_path = agent_nav_path
    # else:
    #     print("Graph is empty, cannot pick random node")
    idx = bs.traf.id2idx("KL001")
    action = bs.traf.hdg[idx] + 1 * 90
    i=0
    #bs.stack.stack(f"HDG KL001 {action}")
    heading_action = 0  # -1 for left, 0 for straight, 1 for right
    altitude_action = 4
    while True:
        #print(f"Reward: {reward}")
       
        i=i+1
        # input the action using arrow keys increase or decrease heading by 10 degrees
        
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    heading_action = -1
                elif event.key == pygame.K_RIGHT:
                    heading_action = 1
                elif event.key == pygame.K_UP:
                    altitude_action = min(altitude_action + 1, 2*ALTITUDE_STEPS)
                elif event.key == pygame.K_DOWN:
                    altitude_action = max(altitude_action - 1, 0)
                elif event.key == pygame.K_r:
                    print("Resetting environment.")
                    env.reset()
            elif event.type == pygame.KEYUP:
                if event.key in [pygame.K_LEFT, pygame.K_RIGHT]:
                    heading_action = 0
                    
        _, reward, terminated, _, _ = env.step([heading_action, altitude_action])
        print(reward)
        if terminated:
            print("Episode terminated, resetting environment.")
            env.reset()