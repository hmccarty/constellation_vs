from sim import Sim
from cv2 import ORB_create, drawKeypoints
import numpy as np

# Global defs
sim = Sim(headless=True)


def calc_distances(p0, points):
    return ((p0 - points)**2).sum(axis=1)


def min_subset(pts, K):
    farthest_pts = np.zeros((K, 3))
    farthest_idxs = []
    farthest_pts[0] = pts[np.random.randint(len(pts))]
    distances = calc_distances(farthest_pts[0], pts)
    for i in range(1, K):
        farthest_idxs.append(np.argmax(distances))
        farthest_pts[i] = pts[farthest_idxs[-1]]
        distances = np.minimum(distances, calc_distances(farthest_pts[i], pts))
    return farthest_idxs


# Receive frames
rgb, depth = sim.step()

# Extract features
kps, _ = ORB_create().detectAndCompute(rgb, None)
kps = kps[-10:]
rgb_all_feats = rgb.copy()
drawKeypoints(rgb_all_feats, kps, None, (255, 0, 0), 4)

# Select features


# Hash each basis

# Frame and store points

# Receive new frame

# Extract features and convert

# For each basis

# Frame points and compare to hash table
