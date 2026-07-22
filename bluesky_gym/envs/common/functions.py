import numpy as np
from typing import List
import numpy as np

import networkx as nx

def bound_angle_positive_negative_180(angle_deg: float) -> float:
    """ maps any angle in degrees to the [-180,180] interval 
    Parameters
    __________
    angle_deg: float
        angle that needs to be mapped (in degrees)
    
    Returns
    __________
    angle_deg: float
        input angle mapped to the interval [-180,180] (in degrees)
    """

    if angle_deg > 180:
        return -(360 - angle_deg)
    elif angle_deg < -180:
        return (360 + angle_deg)
    else:
        return angle_deg

def get_point_at_distance(lat1, lon1, d, bearing, R=6371):
    """
    lat: latitude of the reference point, in degrees
    lon: longitude of the referemce point, in degrees
    d: target distance from the reference point, in km
    bearing: (true) heading, in degrees
    R: optional radius of sphere, defaults to mean radius of earth

    Returns new lat/lon coordinate {d}km from the reference point, in degrees
    """
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    a = np.radians(bearing)
    lat2 = np.arcsin(np.sin(lat1) * np.cos(d/R) + np.cos(lat1) * np.sin(d/R) * np.cos(a))
    lon2 = lon1 + np.arctan2(
        np.sin(a) * np.sin(d/R) * np.cos(lat1),
        np.cos(d/R) - np.sin(lat1) * np.sin(lat2)
    )
    return np.degrees(lat2), np.degrees(lon2)

def random_point_on_circle(radius: float,generator: np.random.Generator) -> np.array:
    """ Get a random point on a circle circumference with given radius
    Parameters
    __________
    radius: float
        radius for the circle
    
    Returns
    __________
    point: np.array
        randomly sampled point
    """
    alpha = 2 * np.pi * generator.uniform(0., 1.)
    x = radius * np.cos(alpha)
    y = radius * np.sin(alpha)
    return np.array([x, y])


def sort_points_clockwise(vertices: np.array) -> np.array:
    """ Sort the points in clockwise order
    Parameters
    __________
    vertices: np.array
        array of points
    
    Returns
    __________
    sorted_vertices: np.array
        sorted array of points
    """
    sorted_vertices = [vertices[i] for i in np.argsort([np.arctan2(v[1], v[0]) for v in vertices])]

    return sorted_vertices   

def polygon_area(vertices: np.array) -> float:
    """ Calculate the area of a polygon given the vertices
    Parameters
    __________
    vertices: np.array
        array of vertices of the polygon
    
    Returns
    __________
    area: float
        area of the polygon
    """
    n = len(vertices)
    area = 0.0
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]  # Wrap around to the first vertex
        area += x1 * y2 - y1 * x2
    area = np.abs(area) / 2.0
    return area

def nm_to_latlong(center: np.array, point: np.array) -> np.array:
    """ Convert a point in nm to lat/long coordinates
    Parameters
    __________
    center: np.array
        center point of the conversion
    point: np.array
        point to be converted
    
    Returns
    __________
    latlong: np.array
        converted point in lat/long coordinates
    """
    lat = center[0] + (point[0] / 60)
    lon = center[1] + (point[1] / (60 * np.cos(np.radians(center[0]))))
    return np.array([lat, lon])

def latlong_to_nm(center: np.array, point: np.array) -> np.array:
    """ Convert a point in lat/long coordinates to nm
    Parameters
    __________
    center: np.array
        center point of the conversion
    point: np.array
        point to be converted
    
    Returns
    __________
    nm: np.array
        converted point in nm
    """
    x = (point[0] - center[0]) * 60
    y = (point[1] - center[1]) * 60 * np.cos(np.radians(center[0]))
    return np.array([x, y])

def euclidean_distance(point1: np.array, point2: np.array) -> float:
    """ Calculate the euclidean distance between two points
    Parameters
    __________
    point1: np.array
        [x, y] of the first point
    point2: np.array
        [x, y] of the second point
        
    Returns
    __________
    distance: float
        euclidean distance between the two points
    """
    return np.sqrt(np.sum((point2 - point1)**2))

def get_hdg(point1: np.array, point2: np.array) -> float:
    """ Calculate the heading from point1 to point2
    Parameters
    __________
    point1: np.array
        [lat, lon] of the first point 
    point2: np.array
        [lat, lon] of the second point
    
    Returns
    __________
    hdg: float
        heading from point1 to point2
    """
    
    lat1, lon1 = np.radians(point1)
    lat2, lon2 = np.radians(point2)
    
    delta_lon = lon2 - lon1
    
    x = np.sin(delta_lon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(delta_lon)
    
    hdg = np.degrees(np.arctan2(x, y))
    
    hdg = (hdg + 360) % 360 # Convert back to [0, 360] interval
    
    return hdg


def polygon_to_segments(vertices: np.array) -> np.ndarray:
    """Convert polygon vertices to an array of line segments.

    Parameters
    __________
    vertices: np.array
        Iterable of 2D points [[x1,y1], [x2,y2], ...]

    Returns
    __________
    segments: np.ndarray
        Array of shape (N,4) where each row is [x1, y1, x2, y2]
    """
    vs = np.asarray(vertices, dtype=float)
    if vs.ndim != 2 or vs.shape[1] != 2:
        raise ValueError("vertices must be an (N,2) array-like of points")

    p1 = vs
    p2 = np.roll(vs, -1, axis=0)
    segments = np.hstack((p1, p2))
    return segments


def segments_intersection_matrix(segs1: np.ndarray, segs2: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Vectorized check for intersections between two sets of segments.

    Parameters
    __________
    segs1: np.ndarray
        (M,4) array rows [x1,y1,x2,y2]
    segs2: np.ndarray
        (N,4) array rows [x1,y1,x2,y2]
    eps: float
        numerical tolerance for parallel/colinear checks

    Returns
    __________
    intersects: np.ndarray
        boolean array of shape (M, N) where True means the corresponding
        segment pair intersects (including colinear overlapping).
    """
    s1 = np.asarray(segs1, dtype=float)
    s2 = np.asarray(segs2, dtype=float)
    if s1.ndim != 2 or s1.shape[1] != 4:
        raise ValueError("segs1 must be (M,4) array")
    if s2.ndim != 2 or s2.shape[1] != 4:
        raise ValueError("segs2 must be (N,4) array")

    # p + t*r  and  q + u*s
    p = s1[:, :2][:, None, :]            # (M,1,2)
    r = (s1[:, 2:] - s1[:, :2])[:, None, :]  # (M,1,2)
    q = s2[:, :2][None, :, :]            # (1,N,2)
    s = (s2[:, 2:] - s2[:, :2])[None, :, :]  # (1,N,2)

    def cross(a, b):
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

    qp = q - p  # (M,N,2)
    denom = cross(r, s)  # (M,N)

    # general case: denom != 0
    t = cross(qp, s) / (denom + (denom == 0) * 1.0)  # avoid div-by-zero
    u = cross(qp, r) / (denom + (denom == 0) * 1.0)

    intersects = np.zeros(denom.shape, dtype=bool)

    non_parallel = np.abs(denom) > eps
    if np.any(non_parallel):
        t_np = t[non_parallel]
        u_np = u[non_parallel]
        hits = (t_np >= -eps) & (t_np <= 1 + eps) & (u_np >= -eps) & (u_np <= 1 + eps)
        intersects[non_parallel] = hits

    # parallel or colinear cases: denom ~= 0
    parallel = ~non_parallel
    if np.any(parallel):
        # Check colinearity: cross(q-p, r) == 0
        col = np.abs(cross(qp, r))[parallel] <= eps
        if np.any(col):
            # For colinear: check 1D projection overlap
            # project onto r (segment direction of segs1)
            # compute dot products for scalars along r
            p_par = p[parallel][col]  # (K,1,2)
            r_par = r[parallel][col]  # (K,1,2)
            q_par = q[parallel][col]  # (K,1,2)
            s_par = s[parallel][col]  # (K,1,2)

            rr = np.sum(r_par[..., :] * r_par[..., :], axis=-1)  # (K,1)
            # avoid zero-length segments
            rr_safe = rr.copy()
            rr_safe[rr_safe == 0] = eps

            t0 = np.sum((q_par - p_par) * r_par, axis=-1) / rr_safe
            t1 = t0 + np.sum(s_par * r_par, axis=-1) / rr_safe

            tmin = np.minimum(t0, t1)
            tmax = np.maximum(t0, t1)

            overlap = (tmax >= -eps) & (tmin <= 1 + eps)

            # assign overlap results back into intersects
            inds = np.argwhere(parallel)
            col_inds = inds[col]
            for idx, val in zip(col_inds, overlap):
                intersects[tuple(idx)] = bool(val)

    return intersects


def segments_intersect_any(segs: np.ndarray, others: np.ndarray) -> np.ndarray:
    """Return boolean vector of length len(segs) indicating if each segment
    intersects any of the `others` segments."""
    mat = segments_intersection_matrix(segs, others)
    return np.any(mat, axis=1)


def segments_intersection_params(segs1: np.ndarray, segs2: np.ndarray, eps: float = 1e-9):
    """Compute parametric intersection values between two segment sets.

    Returns (t, u, intersects) where t and u are arrays of shape (M,N)
    containing the parametric location along segs1 and segs2 respectively
    where an intersection occurs. Non-intersecting pairs contain `np.nan`.
    """
    s1 = np.asarray(segs1, dtype=float)
    s2 = np.asarray(segs2, dtype=float)
    if s1.ndim != 2 or s1.shape[1] != 4:
        raise ValueError("segs1 must be (M,4) array")
    if s2.ndim != 2 or s2.shape[1] != 4:
        raise ValueError("segs2 must be (N,4) array")

    p = s1[:, :2][:, None, :]            # (M,1,2)
    r = (s1[:, 2:] - s1[:, :2])[:, None, :]  # (M,1,2)
    q = s2[:, :2][None, :, :]            # (1,N,2)
    s = (s2[:, 2:] - s2[:, :2])[None, :, :]  # (1,N,2)

    def cross(a, b):
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

    qp = q - p  # (M,N,2)
    denom = cross(r, s)  # (M,N)

    t = np.full(denom.shape, np.nan, dtype=float)
    u = np.full(denom.shape, np.nan, dtype=float)
    intersects = np.zeros(denom.shape, dtype=bool)

    non_parallel = np.abs(denom) > eps
    if np.any(non_parallel):
        t_np = cross(qp, s)[non_parallel] / denom[non_parallel]
        u_np = cross(qp, r)[non_parallel] / denom[non_parallel]
        t[non_parallel] = t_np
        u[non_parallel] = u_np
        hits = (t_np >= -eps) & (t_np <= 1 + eps) & (u_np >= -eps) & (u_np <= 1 + eps)
        intersects[non_parallel] = hits

    # parallel or colinear
    parallel = ~non_parallel
    if np.any(parallel):
        # check colinearity
        col = np.abs(cross(qp, r))[parallel] <= eps
        if np.any(col):
            p_par = p[parallel][col]
            r_par = r[parallel][col]
            q_par = q[parallel][col]
            s_par = s[parallel][col]

            rr = np.sum(r_par * r_par, axis=-1)
            rr_safe = rr.copy()
            rr_safe[rr_safe == 0] = eps

            t0 = np.sum((q_par - p_par) * r_par, axis=-1) / rr_safe
            t1 = t0 + np.sum(s_par * r_par, axis=-1) / rr_safe

            tmin = np.minimum(t0, t1)
            tmax = np.maximum(t0, t1)

            overlap = (tmax >= -eps) & (tmin <= 1 + eps)

            inds = np.argwhere(parallel)
            col_inds = inds[col]
            for (i0, i1), ov, tt0, tt1 in zip(col_inds, overlap, tmin, tmax):
                if ov:
                    intersects[i0, i1] = True
                    # provide a representative t value (clamped within [0,1])
                    t_val = np.minimum(1.0, np.maximum(0.0, tt0))
                    t[i0, i1] = t_val
                    # compute corresponding u by projecting onto seg2 if possible
                    # fallback: leave u as nan

    return t, u, intersects

def load_graph(vertices_path, edges_path):
    """Load a navigation graph from CSV files.

    Parameters
    ----------
    vertices_path : str
        Path to the vertices CSV file (columns: id, lat, lon, altitude, is_airport).
    edges_path : str
        Path to the edges CSV file (columns: node1, node2, distance).

    Returns
    -------
    networkx.Graph
        Graph with node attributes 'lat' and 'lon', and edge attribute 'weight'.
    """
    graph = nx.Graph()
    with open(vertices_path, 'r') as f:
        f.readline()  # skip header
        for line in f:
            parts = line.strip().split(',')
            node_id = str(parts[0])
            lat = float(parts[1])
            lon = float(parts[2])
            altitude = float(parts[3])
            is_airport = int(parts[4])
            if not is_airport:
                graph.add_node(node_id, lat=lat, lon=lon)
    with open(edges_path, 'r') as f:
        f.readline()  # skip header
        for line in f:
            parts = line.strip().split(',')
            node1 = str(parts[0])
            node2 = str(parts[1])
            distance = float(parts[2])
            if node1 in graph.nodes and node2 in graph.nodes:
                graph.add_edge(node1, node2, weight=distance)
    return graph
