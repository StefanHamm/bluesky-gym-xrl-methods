import numpy as np
import pygame
import matplotlib.pyplot as plt
import bluesky as bs
from bluesky_gym.envs.common.screen_dummy import ScreenDummy
import bluesky_gym.envs.common.functions as fn
import math
from scipy.spatial import ConvexHull
import alphashape
from shapely.geometry import Point

import gymnasium as gym
from gymnasium import spaces
import networkx as nx

DISTANCE_MARGIN = 5 # km
REACH_REWARD = 1

AIRWAY_WIDTH = 8 # NM

DRIFT_PENALTY = -0.1
INTRUSION_PENALTY = -1

NUM_INTRUDERS = 5
NUM_WAYPOINTS = 1
INTRUSION_DISTANCE = 5 # NM

MIN_ROUTE_LENGTH = 30 #

WAYPOINT_DISTANCE_MIN = 100
WAYPOINT_DISTANCE_MAX = 150

D_HEADING = 45

AC_SPD = 150

NM2KM = 1.852

ACTION_FREQUENCY = 10

# NAVPOINTS
EDGES_PATH = "data/edges.csv"
VERTICES_PATH = "data/vertices.csv"


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


class NavWaypointEnv(gym.Env):
    """Navpoint gymnasium creating random navpoint encounters

    Args:
        gym (_type_): _description_
    """
    metadata = {"render_modes": ["rgb_array","human"], "render_fps": 120}
    def __init__(self, render_mode=None, window_width=800,window_height=800, stencil_radius_in_km = 100):
        super().__init__()
        
        # load the graph
        # first load the vertices
        self.graph = load_graph(VERTICES_PATH, EDGES_PATH)
        self.window_width = window_width
        self.window_height = window_height
        self.window_size = (window_width, window_height)
        self.window = None
        self.clock = None
        self.agent_nav_path = None
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

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
        
        return subgraph
    
    def _get_agent_nav_path(self,boundary_vertices):
        # select a random start and end point from the boundary vertices
        if len(boundary_vertices) < 2:
            return []
        start, end = np.random.choice(boundary_vertices, size=2, replace=False)
        # find the shortest path between them
        #use networkx shortest path algorithm with weight as distance
        try:
            path = nx.shortest_path(self.subgraph, source=start, target=end, weight='weight')
            return path
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
        
        return x_pos,y_pos
        
            
    def _post_render(self,canvas):
        self.window.blit(canvas, canvas.get_rect())
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])
        
    def _plot_scale(self,canvas):
        # plot a scale of 10 km in the bottom left corner
        scale_length_in_km = 10
        scale_length_in_px = scale_length_in_km * self.px_per_km
        pygame.draw.line(canvas, (0,0,0), (50, self.window_height-50), (50+scale_length_in_px, self.window_height-50), int(0.5*self.px_per_km))
        font = pygame.font.SysFont(None, 24)
        text = font.render(f"{scale_length_in_km} km", True, (0,0,0))
        canvas.blit(text, (50, self.window_height-80))
        return canvas

    
    def _render_frame(self):
        self._pre_render()
        canvas = pygame.Surface(self.window_size)
        canvas.fill((135,206,235))
        
        canvas = self._plot_scale(canvas)
        
        edge_coords = []
        for u, v in self.subgraph.edges():
            pos_u = self.subgraph.nodes[u]
            pos_v = self.subgraph.nodes[v]
            if 'lat' in pos_u and 'lon' in pos_u and 'lat' in pos_v and 'lon' in pos_v:
                x1, y1 = self.lat_lon_to_screen_coordinates(pos_u['lat'], pos_u['lon'])
                x2, y2 = self.lat_lon_to_screen_coordinates(pos_v['lat'], pos_v['lon'])
                edge_coords.append(((int(x1), int(y1)), (int(x2), int(y2))))
        
        # First pass: draw airways
        for start, end in edge_coords:
            pygame.draw.line(canvas, (0,255,0), start, end, int(AIRWAY_WIDTH*NM2KM*self.px_per_km))
            #also draw a cirlce at the start and end of each edge to make it look better
            pygame.draw.circle(canvas, (0,255,0), start, int(AIRWAY_WIDTH/2*NM2KM*self.px_per_km))
            pygame.draw.circle(canvas, (0,255,0), end, int(AIRWAY_WIDTH/2*NM2KM*self.px_per_km))
            
        # Second pass: draw center lines
        for start, end in edge_coords:
            pygame.draw.line(canvas, (0,0,0), start, end, 1)

        for node in self.subgraph.nodes:
            data = self.subgraph.nodes[node]
            if 'lat' in data and 'lon' in data:
                x, y = self.lat_lon_to_screen_coordinates(data['lat'], data['lon'])
                
                color = (255, 0, 0)
                if self.agent_nav_path and node in self.agent_nav_path:
                    color = (255, 255, 0)
                
                pygame.draw.circle(canvas, color, (int(x), int(y)), 3) 
        
        boundary_vertices = self._get_boundary_vertices()
        for node in boundary_vertices:
            data = self.subgraph.nodes[node]
            if 'lat' in data and 'lon' in data:
                x, y = self.lat_lon_to_screen_coordinates(data['lat'], data['lon'])
                
                color = (0, 0, 255)
                if self.agent_nav_path and node in self.agent_nav_path:
                    color = (255, 255, 0)
                
                pygame.draw.circle(canvas, color, (int(x), int(y)), 3)
        
        # plot a sample plane on the center
        # it must be the plane and the intrusion circle with radius of 5 NM
        center_x = int(self.window_width / 2)
        center_y = int(self.window_height / 2)
        
        # Intrusion Circle
        radius_px = int(INTRUSION_DISTANCE * NM2KM * self.px_per_km)
        pygame.draw.circle(canvas, (255, 0, 0), (center_x, center_y), radius_px, 1)

        # Ownship Triangle facing North
        # North is Up (negative Y)
        size = 10
        points = [
            (center_x, center_y - size),           # Top point
            (center_x - size / 1.5, center_y + size), # Bottom left
            (center_x + size / 1.5, center_y + size)  # Bottom right
        ]
        pygame.draw.polygon(canvas, (0, 0, 0), points)
        
        
        self._post_render(canvas)
        



if __name__ == "__main__":
    env = NavWaypointEnv(render_mode="human",window_height=1000,window_width=1000,stencil_radius_in_km=100)
    
    random_node = np.random.choice(list(env.graph.nodes))
    env.center_point = {"lat": env.graph.nodes[random_node]['lat'], "lon": env.graph.nodes[random_node]['lon']}
    subgraph = env._get_subgraph_around_waypoint((env.graph.nodes[random_node]['lat'], env.graph.nodes[random_node]['lon']), 100)
    env.subgraph = subgraph
    #print number of nodes in subgraph
    print(f"Subgraph has {subgraph.number_of_nodes()} nodes and {subgraph.number_of_edges()} edges")
    env.center_point = {"lat": env.graph.nodes[random_node]['lat'], "lon": env.graph.nodes[random_node]['lon']}
    
    boundary_vertices = env._get_boundary_vertices()
    
    agent_nav_path = env._get_agent_nav_path(boundary_vertices)
    env.agent_nav_path = agent_nav_path
    
    
    while True:
        env._render_frame()
        