from sim import Sim
from feature_finder import FeatureFinder
from feature_matcher import FeatureMatcher
from itertools import combinations
import cv2 as cv
import numpy as np
import time
from ibvs import IBVS

VOTE_THRESHOLD = 3

CAM_WIDTH = 450  # pixels
CAM_HEIGHT = 450  # pixels
CAM_CLOSEST = 0.1  # meters
CAM_FARTHEST = 8.0  # meters
CONSTELLATION_SIZE = 6  # number of points
X_HASH_SIZE = 0.001  # meters
Y_HASH_SIZE = 0.001  # meters
Z_HASH_SIZE = 0.01  # meters

sim = Sim(headless=True)
finder = FeatureFinder()

# Intel camera configuration:
# ibvs = IBVS(np.array([800, 450]), 1280, 1204, 0.0000014)

# Sim camera configuration:
ibvs = IBVS(
    np.array([CAM_WIDTH / 2., CAM_HEIGHT / 2.]), 543.19, 543.19, 0.001)
start = time.time()
prev_root_pnt = None

# Preset visual-servoing command
target_pxl = np.array([230., 300.])
goal_pxl = np.array([CAM_WIDTH / 2., CAM_HEIGHT / 2.])

# Number of reference frames stored
num_frames = 0

# Holds features with desired selection criteria
feats = None

# Dictionary of framed points
constellation = None

min_pnt = ibvs.feature_to_pnt(np.array([0, 0]), np.full(
    (CAM_HEIGHT + 1, CAM_WIDTH + 1), CAM_FARTHEST))
min_pnt[2] = -CAM_CLOSEST
max_pnt = ibvs.feature_to_pnt(np.array([CAM_WIDTH, CAM_HEIGHT]), np.full(
    (CAM_HEIGHT + 1, CAM_WIDTH + 1), CAM_FARTHEST))
pnt_range = max_pnt - min_pnt
x_size = int(pnt_range[0] / X_HASH_SIZE)
y_size = int(pnt_range[1] / Y_HASH_SIZE)
z_size = int((CAM_FARTHEST - CAM_CLOSEST) / Z_HASH_SIZE)


def store(constellation, pnt, ref_frame, id):
    idx = pnt - min_pnt
    x_idx = int(idx[0] / X_HASH_SIZE)
    y_idx = int(idx[1] / Y_HASH_SIZE)
    z_idx = int(idx[2] / Z_HASH_SIZE)
    if x_idx >= 0 and x_idx < x_size \
            and y_idx >= 0 and y_idx < y_size \
            and z_idx >= 0 and z_idx < z_size:
        if constellation[x_idx][y_idx][z_idx] is None:
            constellation[x_idx][y_idx][z_idx] = [(ref_frame, id)]
        else:
            constellation[x_idx][y_idx][z_idx].append((ref_frame, id))


def get_frames(constellation, pnt):
    idx = pnt - min_pnt
    x_idx = int(idx[0] / X_HASH_SIZE)
    y_idx = int(idx[1] / Y_HASH_SIZE)
    z_idx = int(idx[2] / Z_HASH_SIZE)
    if x_idx >= 0 and x_idx < x_size \
            and y_idx >= 0 and y_idx < y_size \
            and z_idx >= 0 and z_idx < z_size:
        return constellation[x_idx][y_idx][z_idx]


def removearray(L, arr):
    ind = 0
    size = len(L)
    while ind != size and not np.array_equal(L[ind], arr):
        ind += 1
    if ind != size:
        L.pop(ind)
    else:
        raise ValueError('array not found in list.')


# Perform VS for 15 seconds
vel = np.array([0.1, 0., 0., 0., 0., 0.])
while (time.time() - start) < 10:
    # Get image from sim and find ORB keypoints
    rgb, depth = sim.step()
    featimg, kps, des = finder.get_orb(rgb)

    # Convert target and goal pixels to cartesian space
    target_pnt = ibvs.feature_to_pnt(target_pxl, depth)
    goal_pnt = ibvs.feature_to_pnt(goal_pxl, depth)

    if constellation is None:
        #                         #
        #   Feature selection     #
        #                         #
        # Find translational distance from goal
        diff = goal_pnt - target_pnt

        constellation = [[[None] * z_size] * y_size] * x_size

        # TODO: Select features by score and distance
        feats = []
        pnts = []

        for i in range(CONSTELLATION_SIZE):
            feat = np.array(kps[i].pt)
            feats.append(feat)
            pnts.append(ibvs.feature_to_pnt(feat, depth))

        #                           #
        #   Frame constellation     #
        #                           #
        frame_combos = combinations(pnts, 3)
        for frame_pnts in frame_combos:
            frame = np.array(frame_pnts).T

            # Calculate remaining points using ref frame
            rem_pnts = list(pnts)
            for pnt in frame_pnts:
                removearray(rem_pnts, pnt)
            rem_pnts = np.array(rem_pnts).T

            framed_pnts = np.linalg.solve(frame, rem_pnts).T

            # Add framed points to constellation
            for pnt in framed_pnts:
                store(constellation, pnt, frame, num_frames)

            num_frames += 1

    #                           #
    #   Match constellation     #
    #                           #
    else:
        # Get all features
        feats = []
        pnts = []

        for kp in kps:
            pnts.append(ibvs.feature_to_pnt(np.array(kp.pt), depth))

        frame_combos = combinations(pnts, 3)
        max_votes = 0
        for frame_pnts in frame_combos:
            votes = [[0, []]] * num_frames
            frame = np.array(frame_pnts).T
            try:
                world = np.linalg.inv(frame)
            except np.linalg.LinAlgError:
                break

            # Calculate remaining points using ref frame
            rem_pnts = list(pnts)
            for pnt in frame_pnts:
                removearray(rem_pnts, pnt)
            rem_pnts = np.array(rem_pnts).T

            framed_pnts = np.linalg.solve(frame, rem_pnts).T

            # Add framed points to constellation
            for pnt in framed_pnts:
                frames = get_frames(constellation, pnt)
                if frames is not None:
                    pnt = np.linalg.solve(world, pnt)
                    for frame in frames:
                        votes[frame[1]][0] += 1
                        votes[frame[1]][1].append(ibvs.pnt_to_feature(pnt))

            for vote in votes:
                if vote[0] > max_votes:
                    max_votes = vote[0]
                    feats = vote[1]

    #              #
    #   Update     #
    #              #

    if constellation is not None:
        # Use velocity command to converge goal to target
        sim.applyVelocity(vel)

        # Draw constellation to output image
        for pxl in feats:
            pos = (int(pxl[0]), int(pxl[1]))
            cv.circle(featimg, pos, 5, (0, 255, 0), -1)

        # Print debug info on constellation
        # for i in range(x_size):
        #     for j in range(y_size):
        #         for k in range(z_size):
        #             if constellation[i][j][k] is not None:
        #                 print("Bin at {}, {}, {} has entry".format(i, j, k))

        # for pxl in feats:
        #     begin = ibvs.pnt_to_feature(root_pnt)
        #     end = (int(pxl[0]), int(pxl[1]))
        #     cv.line(featimg, tuple(begin.astype(int)),
        #             end, (0, 255, 0), 5)

        # for pnt in constellation:
        #     begin = ibvs.pnt_to_feature(root_pnt)
        #     end = ibvs.pnt_to_feature(pnt + root_pnt)
        #     cv.line(featimg, tuple(begin.astype(int)),
        #             tuple(end.astype(int)), (0, 255, 0), 5)

        # Draw expected location of constellation to output image
        # for pxl in ibvs._goal:
        #     pos = (int(pxl[0]), int(pxl[1]))
        #     cv.circle(featimg, pos, 5, (0, 125, 125), -1)

        # if len(feats == 4):
        #     # Update velocity command
        #     vel = ibvs.execute(feats, depth)

        #     # Use velocity command to converge goal to target
        #     sim.applyVelocity(vel)

        # if prev_root_pnt is not None:
        #     target_pnt += (root_pnt - prev_root_pnt)
        # prev_root_pnt = root_pnt
    else:
        prev_root_pnt = None

    # # Draw target
    # target_pxl = ibvs.pnt_to_feature(target_pnt)
    # cv.circle(featimg, tuple(target_pxl.astype(int)), 15, (0, 0, 255), -1)

    # # Draw goal
    # cv.circle(featimg, tuple(goal_pxl.astype(int)), 15, (125, 125, 0), -1)

    # Show image and sleep for sim
    cv.imshow("Output Image", featimg)
    cv.waitKey()
    time.sleep(sim.dt)

cv.destroyAllWindows()
sim.disconnect()
