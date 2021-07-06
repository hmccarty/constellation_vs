from numpy.linalg.linalg import _is_empty_2d
from sim import Sim
from feature_finder import FeatureFinder
from geo_hasher import GeoHasher
from itertools import permutations, combinations
import cv2 as cv
import numpy as np
import time
from ibvs import IBVS
from util import rigid_tf
from scipy.spatial.distance import cdist

VOTE_THRESHOLD = 5

CAM_WIDTH = 450  # pixels
CAM_HEIGHT = 450  # pixels
CAM_CLOSEST = 0.1  # meters
CAM_FARTHEST = 8.0  # meters
CONSTELLATION_SIZE = 5  # number of points
X_HASH_SIZE = 15.  # meters
Y_HASH_SIZE = 15.  # meters
Z_HASH_SIZE = 1.  # meters

sim = Sim(headless=True)
finder = FeatureFinder()

# Intel camera configuration:
# ibvs = IBVS(np.array([800, 450]), 1280, 1204, 0.0000014)

# Sim camera configuration:
ibvs = IBVS(
    np.array([CAM_WIDTH / 2., CAM_HEIGHT / 2.]), 543.19, 543.19, 0.001)
start = time.time()
prev_root_pnt = None

# Geometric hashing for constellation
constellation = GeoHasher((X_HASH_SIZE, Y_HASH_SIZE, Z_HASH_SIZE),
                          VOTE_THRESHOLD,
                          CONSTELLATION_SIZE)

# Preset visual-servoing command
target_pxl = np.array([230., 300.])
goal_pxl = np.array([CAM_WIDTH / 2., CAM_HEIGHT / 2.])

# Holds features with desired selection criteria
pnts = None
prev_pnts = None
feats = None
draw = []

# Perform VS for t seconds
t = 50
vel = np.array([0., 0., 0., 0., 0., 0.])
prev_time = start
while (time.time() - start) < t:
    # Get image from sim and find ORB keypoints
    print("Starting new frame")
    rgb, depth = sim.step()
    featimg, kps, des = finder.get_orb(rgb)

    # Convert target and goal pixels to cartesian space
    target_pnt = ibvs.feature_to_pnt(target_pxl, depth)
    goal_pnt = ibvs.feature_to_pnt(goal_pxl, depth)

    if constellation.is_empty():
        #                         #
        #   Feature selection     #
        #                         #
        # Find translational distance from goal
        diff = goal_pnt - target_pnt

        # TODO: Select features by score and distance
        feats = []
        pnts = []
        dist = None

        for i in range(0, len(kps)):
            feat = np.array(kps[i].pt)
            feats.append(feat)
            pnts.append(ibvs.feature_to_pnt(feat, depth))

            if dist is None:
                dist = np.array([pnts[-1]])
            else:
                dist = np.vstack((dist, pnts[-1]))

        # print(cdist(dist, dist))
        dist = cdist(dist, dist)
        dist = np.square(np.triu(dist))
        dist = np.mean(np.divide(1, dist, where=dist != 0), axis=1)
        # print(dist)
        idxs = dist.argsort()[:CONSTELLATION_SIZE][::-1]

        # Sort points to find those within maximize min-distance
        idxs = dist.argsort()[::-1][:CONSTELLATION_SIZE]
        new_pnts = []
        new_feats = []
        for i in idxs:
            new_pnts.append(pnts[i])
            new_feats.append(feats[i])

        pnts = new_pnts
        feats = new_feats

        #                           #
        #   Frame constellation     #
        #                           #
        for frame_idxs in permutations(range(len(pnts)), 3):
            # Calculate frame
            frame_pnts = []
            for idx in frame_idxs:
                frame_pnts.append(pnts[idx])
            origin, frame = constellation.calculate_frame(frame_pnts)

            if frame is not None:
                # Remove frame pnts from remaining pnts
                rem_pnts = np.array(pnts).T

                # Frame and store remaining points
                framed_pnts = constellation.store(origin, frame, rem_pnts)

        ibvs.set_goal(feats, depth, diff)
    #                           #
    #   Match constellation     #
    #                           #
    else:
        # Get all features
        feats = []
        pnts = []
        matched_frame = []
        matched_pnts = []
        matched_cnt = 0

        for kp in kps:
            pnts.append(ibvs.feature_to_pnt(np.array(kp.pt), depth))

        for frame_idxs in permutations(range(len(pnts)), 3):
            # Calculate frame
            frame_pnts = []
            for idx in frame_idxs:
                frame_pnts.append(pnts[idx])
            origin, frame = constellation.calculate_frame(frame_pnts)

            if frame is not None:
                # Remove frame pnts from remaining pnts
                rem_pnts = np.array(pnts).T

                possible_pnts, cnt = constellation.vote(
                    origin, frame, rem_pnts)
                if possible_pnts is not None and \
                        cnt > matched_cnt:
                    matched_pnts = possible_pnts
                    matched_cnt = cnt

            if matched_cnt >= VOTE_THRESHOLD:
                break

        if matched_pnts is not None:
            pnts = matched_pnts
            for pnt in matched_pnts:
                if pnt is not None:
                    feats.append(ibvs.pnt_to_feature(pnt))

    #              #
    #   Update     #
    #              #

    # # Draw goal
    cv.circle(featimg, tuple(goal_pxl.astype(int)), 15, (125, 125, 0), -1)

    if not constellation.is_empty():
        print("Number of features used: {}".format(len(feats)))

        # Draw constellation to output image
        for pxl in feats:
            pos = (int(pxl[0]), int(pxl[1]))
            cv.circle(featimg, pos, 5, (0, 255, 0), -1)

        if len(feats) == CONSTELLATION_SIZE:
            vel = ibvs.execute(feats, depth)
            if prev_pnts is not None:
                R, trans = rigid_tf(prev_pnts, pnts)
                target_pnt = target_pnt + trans.T[0]
                print("Found translation: ", trans)
        sim.applyVelocity(vel)

    # Draw target
    target_pxl = ibvs.pnt_to_feature(target_pnt)
    cv.circle(featimg, tuple(target_pxl.astype(int)), 15, (0, 0, 255), -1)

    prev_pnts = pnts

    # now = time.time()
    # print("Time taken: {}".format(now - prev_time))
    # prev_time = now

    # Show image and sleep for sim
    cv.imshow("Output Image", featimg)
    print("Frame finished")
    cv.waitKey()

cv.destroyAllWindows()
sim.disconnect()
