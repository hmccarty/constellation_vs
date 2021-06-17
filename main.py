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

VOTE_THRESHOLD = 5

CAM_WIDTH = 450  # pixels
CAM_HEIGHT = 450  # pixels
CAM_CLOSEST = 0.1  # meters
CAM_FARTHEST = 8.0  # meters
CONSTELLATION_SIZE = 5  # number of points
<<<<<<< HEAD
X_HASH_SIZE = 7.0  # meters
Y_HASH_SIZE = 7.0  # meters
=======
X_HASH_SIZE = 15.  # meters
Y_HASH_SIZE = 15.  # meters
>>>>>>> 6b98ccc2e422bdc8ec9cb0d6139c1b3df35ec17d
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
<<<<<<< HEAD
constellation = GeoHasher((X_HASH_SIZE, Y_HASH_SIZE, Z_HASH_SIZE),
                          VOTE_THRESHOLD,
                          CONSTELLATION_SIZE)
=======
# constellation = GeoHasher(
#     (X_HASH_SIZE, Y_HASH_SIZE, Z_HASH_SIZE), VOTE_THRESHOLD)
constellation = GeoHasher(
    (X_HASH_SIZE, Y_HASH_SIZE), VOTE_THRESHOLD)
>>>>>>> 6b98ccc2e422bdc8ec9cb0d6139c1b3df35ec17d

# Preset visual-servoing command
target_pxl = np.array([230., 300.])
goal_pxl = np.array([CAM_WIDTH / 2., CAM_HEIGHT / 2.])

# Holds features with desired selection criteria
<<<<<<< HEAD
pnts = None
prev_pnts = None
feats = None
draw = []

# Perform VS for t seconds
t = 5
vel = np.array([0., 0., 0., 0., 0., 0.])
prev_time = start
while (time.time() - start) < t:
=======
feats = None
draw = []

# Perform VS for 15 seconds
vel = np.array([0.0, 0., 0., 0., 0., 0.])
while (time.time() - start) < 10:
>>>>>>> 6b98ccc2e422bdc8ec9cb0d6139c1b3df35ec17d
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

        for i in range(0, len(kps), int(len(kps) / CONSTELLATION_SIZE)):
            feat = np.array(kps[i].pt)
            feats.append(feat)
<<<<<<< HEAD
            pnts.append(ibvs.feature_to_pnt(feat, depth))
=======
            # pnts.append(ibvs.feature_to_pnt(feat, depth))
            pnts.append(feat)
>>>>>>> 6b98ccc2e422bdc8ec9cb0d6139c1b3df35ec17d

        #                           #
        #   Frame constellation     #
        #                           #
<<<<<<< HEAD
=======
        print(pnts)
>>>>>>> 6b98ccc2e422bdc8ec9cb0d6139c1b3df35ec17d
        for frame_idxs in permutations(range(len(pnts)), 3):
            # Calculate frame
            frame_pnts = []
            for idx in frame_idxs:
                frame_pnts.append(pnts[idx])
            origin, frame = constellation.calculate_frame(frame_pnts)

            if frame is not None:
                # Remove frame pnts from remaining pnts
                rem_pnts = np.array(pnts).T
<<<<<<< HEAD

                # Frame and store remaining points
                framed_pnts = constellation.store(origin, frame, rem_pnts)

        ibvs.set_goal(feats, depth, diff)
=======
                # rem_pnts = np.delete(rem_pnts, frame_idxs, 1)

                # Frame and store remaining points
                framed_pnts = constellation.store(origin, frame, rem_pnts)
                print(framed_pnts)
                for i in range(len(framed_pnts)):
                    draw.append(framed_pnts[i])
                    # draw.append(ibvs.pnt_to_feature(framed_pnts[i]))

                draw2 = []
                for i in range(len(frame_pnts)):
                    draw2.append(frame_pnts[i])
                    # draw2.append(ibvs.pnt_to_feature(frame_pnts[i]))

>>>>>>> 6b98ccc2e422bdc8ec9cb0d6139c1b3df35ec17d
    #                           #
    #   Match constellation     #
    #                           #
    else:
        # Get all features
        feats = []
        pnts = []
        matched_frame = []
        matched_pnts = []
<<<<<<< HEAD

        for kp in kps:
            pnts.append(ibvs.feature_to_pnt(np.array(kp.pt), depth))

        for frame_idxs in combinations(range(len(pnts)), 3):
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
                        cnt > len(matched_pnts):
                    matched_pnts = possible_pnts

            if len(matched_pnts) >= VOTE_THRESHOLD:
                break

        if matched_pnts is not None:
            pnts = matched_pnts
            for pnt in matched_pnts:
                if pnt is not None:
                    feats.append(ibvs.pnt_to_feature(pnt))

    #              #
    #   Update     #
    #              #

    # Draw target
    target_pxl = ibvs.pnt_to_feature(target_pnt)
    cv.circle(featimg, tuple(target_pxl.astype(int)), 15, (0, 0, 255), -1)

    # Draw goal
    cv.circle(featimg, tuple(goal_pxl.astype(int)), 15, (125, 125, 0), -1)

    if not constellation.is_empty():
        print("Number of features used: {}".format(len(feats)))
=======

        for kp in kps:
            pnts.append(np.array(kp.pt))
            # pnts.append(ibvs.feature_to_pnt(np.array(kp.pt), depth))

        for frame_idxs in permutations(range(len(pnts)), 3):
            # Calculate frame
            frame_pnts = []
            for idx in frame_idxs:
                frame_pnts.append(pnts[idx])
            origin, frame = constellation.calculate_frame(frame_pnts)

            if frame is not None:
                # Remove frame pnts from remaining pnts
                rem_pnts = np.array(pnts).T
                # rem_pnts = np.delete(rem_pnts, frame_idxs, 1)

                possible_pnts, framed_pnts = constellation.vote(
                    origin, frame, rem_pnts)
                if possible_pnts is not None and \
                        len(possible_pnts) > len(matched_pnts):
                    matched_pnts = possible_pnts

                if framed_pnts is not None:
                    draw3 = []
                    for i in range(len(framed_pnts)):
                        # draw3.append(ibvs.pnt_to_feature(framed_pnts[i]))
                        draw3.append(framed_pnts[i])

                    for pxl in draw3:
                        pos = (int(pxl[0]), int(pxl[1]))
                        cv.circle(featimg, pos, 8, (0, 60, 255), -1)

        if matched_pnts is not None:
            for pnt in matched_pnts:
                feats.append(pnt)
                # feats.append(ibvs.pnt_to_feature(pnt))

    #              #
    #   Update     #
    #              #

    if constellation is not None:
        print("Number of features used: {}".format(len(feats)))
        print(feats)

        # Use velocity command to converge goal to target
        sim.applyVelocity(vel)
>>>>>>> 6b98ccc2e422bdc8ec9cb0d6139c1b3df35ec17d

        # Draw constellation to output image
        for pxl in feats:
            pos = (int(pxl[0]), int(pxl[1]))
            cv.circle(featimg, pos, 5, (0, 255, 0), -1)

<<<<<<< HEAD
        if len(feats) == CONSTELLATION_SIZE:
            vel = ibvs.execute(feats, depth)
            if prev_pnts is not None:
                print(prev_pnts)
                print(pnts)
                R, t = rigid_tf(prev_pnts, pnts)
                print(t)

        sim.applyVelocity(vel)

    prev_pnts = pnts

    # now = time.time()
    # print("Time taken: {}".format(now - prev_time))
    # prev_time = now

=======
        for pxl in draw:
            pos = (int(pxl[0]), int(pxl[1]))
            cv.circle(featimg, pos, 5, (125, 125, 0), -1)

        for pxl in draw2:
            pos = (int(pxl[0]), int(pxl[1]))
            cv.circle(featimg, pos, 5, (0, 125, 125), -1)

>>>>>>> 6b98ccc2e422bdc8ec9cb0d6139c1b3df35ec17d
    # Show image and sleep for sim
    cv.imshow("Output Image", featimg)
    print("Frame finished")
    cv.waitKey()

cv.destroyAllWindows()
sim.disconnect()
