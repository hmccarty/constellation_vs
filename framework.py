from sim import Sim
from cv2 import ORB_create, drawKeypoints

sim = Sim(headless=True)

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
