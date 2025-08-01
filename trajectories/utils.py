import numpy as np

def find_nearest_point(path, current_pos):
    diffs = path[:, :2] - current_pos
    dists = np.linalg.norm(diffs, axis=1)
    return np.argmin(dists)

def find_lookahead_point(path, current_pos, lookahead_distance):
    dists = np.linalg.norm(path[:, :2] - current_pos, axis=1)
    ahead_points = np.where(dists >= lookahead_distance)[0]
    return ahead_points[0] if len(ahead_points) > 0 else len(path) - 1
