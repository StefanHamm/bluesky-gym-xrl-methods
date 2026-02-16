"""
Raster-based obstacle generation for airway corridor environments.

Rasterizes corridors onto an offscreen buffer, flood-fills from the border
to find exterior space, then extracts interior void regions as obstacle
polygons in lat/lon coordinates.
"""

import numpy as np
import pygame
import matplotlib.path
import bluesky as bs
from scipy import ndimage


NM2KM = 1.852


class ObstacleRasterizer:
    """Generates and manages raster-based void-obstacles from airway corridors.

    Given a subgraph of airway nodes/edges and a geographic center point,
    this class rasterizes the corridors onto an offscreen buffer and
    identifies interior void regions (faces enclosed by airways) as
    obstacle polygons.

    Parameters
    ----------
    raster_res : int
        Resolution of the offscreen rasterisation buffer (default 1000).
    min_void_px_area : int
        Minimum area of a void region in pixels² to keep (default 15).
    max_contour_verts : int
        Maximum number of vertices kept in each simplified contour polygon (default 20).
    """

    def __init__(self, raster_res=1000, min_void_px_area=15, max_contour_verts=20):
        self.raster_res = raster_res
        self.min_void_px_area = min_void_px_area
        self.max_contour_verts = max_contour_verts

        # Populated after generate() is called
        self.obstacles = []
        self.obstacle_labeled = None
        self.interior_labels = set()

    # --- pixel ↔ lat/lon helpers ------------------------------------------

    @staticmethod
    def latlon_to_px(lat, lon, center_point, stencil_radius_in_km, buf_w, buf_h):
        """Project lat/lon → pixel (x, y) on the raster buffer."""
        qdr, dis = bs.tools.geo.kwikqdrdist(
            center_point['lat'], center_point['lon'], lat, lon)
        frac = (dis * NM2KM) / stencil_radius_in_km
        px_x = buf_w / 2 + np.sin(np.deg2rad(qdr)) * frac * buf_w / 2
        px_y = buf_h / 2 - np.cos(np.deg2rad(qdr)) * frac * buf_h / 2
        return int(round(px_x)), int(round(px_y))

    @staticmethod
    def px_to_latlon(px_x, px_y, center_point, stencil_radius_in_km, buf_w, buf_h):
        """Reverse-project pixel (x, y) → (lat, lon)."""
        dx = (px_x - buf_w / 2) / (buf_w / 2) * stencil_radius_in_km
        dy = -(px_y - buf_h / 2) / (buf_h / 2) * stencil_radius_in_km
        dist_km = np.sqrt(dx**2 + dy**2)
        bearing = np.rad2deg(np.arctan2(dx, dy))
        dist_nm = dist_km / NM2KM
        lat, lon = bs.tools.geo.kwikpos(
            center_point['lat'], center_point['lon'],
            bearing, dist_nm)
        return lat, lon

    # --- boundary extraction ------------------------------------------------

    @staticmethod
    def boundary_to_polygon(region_mask, max_verts=48):
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

    def generate(self, subgraph, center_point, stencil_radius_in_km, airway_width_nm, debug=False):
        """Rasterize corridors and extract interior void regions as obstacle polygons.

        Parameters
        ----------
        subgraph : networkx.Graph
            The subgraph containing nodes with 'lat'/'lon' attributes and edges.
        center_point : dict
            ``{'lat': float, 'lon': float}`` – geographic center for projection.
        stencil_radius_in_km : float
            Radius of the stencil area in km.
        airway_width_nm : float
            Total airway corridor width in nautical miles.
        debug : bool
            If True, print diagnostic information.

        Returns
        -------
        list[dict]
            List of obstacle dicts, each containing:
              - ``coords``   – [(lat, lon), …] polygon of the void region
              - ``path``     – matplotlib.path.Path for point-in-polygon tests
              - ``centroid`` – (lat, lon) of the centroid
        """
        self.obstacles = []
        self.obstacle_labeled = None
        self.interior_labels = set()

        if subgraph is None or subgraph.number_of_edges() == 0:
            return []

        # Ensure pygame is initialised (needed for offscreen Surface)
        if not pygame.get_init():
            pygame.init()

        W = H = self.raster_res
        # corridor half-width in pixels
        corridor_half_px = max(1, int(
            (airway_width_nm / 2.0 * NM2KM) /
            stencil_radius_in_km * (W / 2)))
        corridor_px = corridor_half_px * 2 + 1  # full diameter (odd)

        # ---- 1. Rasterize corridors onto a binary mask --------------------
        buf = pygame.Surface((W, H))
        buf.fill((0, 0, 0))

        for u, v in subgraph.edges():
            pu = subgraph.nodes[u]
            pv = subgraph.nodes[v]
            if 'lat' not in pu or 'lat' not in pv:
                continue
            x1, y1 = self.latlon_to_px(pu['lat'], pu['lon'], center_point, stencil_radius_in_km, W, H)
            x2, y2 = self.latlon_to_px(pv['lat'], pv['lon'], center_point, stencil_radius_in_km, W, H)
            pygame.draw.line(buf, (255, 255, 255), (x1, y1), (x2, y2), corridor_px)
            pygame.draw.circle(buf, (255, 255, 255), (x1, y1), corridor_half_px)
            pygame.draw.circle(buf, (255, 255, 255), (x2, y2), corridor_half_px)

        # Convert to numpy mask: 1 = corridor, 0 = empty
        arr = pygame.surfarray.pixels3d(buf)
        mask = (arr[:, :, 0] > 128).astype(np.uint8)
        del arr  # release surface lock

        # ---- 2. Flood-fill exterior from border ---------------------------
        empty_mask = 1 - mask
        labeled, num_features = ndimage.label(empty_mask)

        border_labels = set()
        border_labels.update(labeled[0, :].tolist())
        border_labels.update(labeled[-1, :].tolist())
        border_labels.update(labeled[:, 0].tolist())
        border_labels.update(labeled[:, -1].tolist())
        border_labels.discard(0)

        # ---- 3. Extract interior void regions -----------------------------
        interior_labels = set()
        obstacles = []
        for lbl in range(1, num_features + 1):
            if lbl in border_labels:
                continue

            region_size = np.sum(labeled == lbl)
            if region_size < self.min_void_px_area:
                continue

            interior_labels.add(lbl)

            region_mask = (labeled == lbl)
            poly_px = self.boundary_to_polygon(region_mask, self.max_contour_verts)
            if len(poly_px) < 3:
                continue

            coords_ll = []
            for px_x, px_y in poly_px:
                lat, lon = self.px_to_latlon(px_x, px_y, center_point, stencil_radius_in_km, W, H)
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

        self.obstacle_labeled = labeled
        self.interior_labels = interior_labels
        self.obstacles = obstacles

        if debug:
            print(f"Obstacle generation (raster): {num_features} void "
                  f"components found, {len(border_labels)} exterior, "
                  f"{len(obstacles)} interior obstacle(s) kept.")

        return obstacles

    def check_collision(self, lat, lon, center_point, stencil_radius_in_km):
        """Check if a lat/lon position is inside a void-obstacle region.

        Parameters
        ----------
        lat, lon : float
            Position to check.
        center_point : dict
            ``{'lat': float, 'lon': float}`` – geographic center for projection.
        stencil_radius_in_km : float
            Radius of the stencil area in km.

        Returns
        -------
        bool
            True if the position is inside an obstacle, False otherwise.
        """
        if self.obstacle_labeled is None or not self.interior_labels:
            return False

        W = H = self.raster_res
        px_x, px_y = self.latlon_to_px(lat, lon, center_point, stencil_radius_in_km, W, H)
        if 0 <= px_x < W and 0 <= px_y < H:
            label = self.obstacle_labeled[px_x, px_y]
            if label in self.interior_labels:
                return True
        return False
