from sim import Sim
from feature_finder import FeatureFinder
from feature_matcher import FeatureMatcher
import cv2 as cv
import numpy as np
import time
from ibvs import IBVS

CONSTELLATION_SIZE = 4

sim = Sim(headless=True)
finder = FeatureFinder()

# Intel camera configuration:
# ibvs = IBVS(np.array([800, 450]), 1280, 1204, 0.0000014)

# Sim camera configuration:
ibvs = IBVS(np.array([225, 225]), 543.19, 543.19, 0.001)
start = time.time()
prev_root_pnt = None

# Preset visual-servoing command
target_pxl = np.array([230., 300.])
goal_pxl = np.array([225., 225.])

# Holds features with desired selection criteria
feats = None

# Holds scale-invariant associations of selected features
constellation = None

# Perform VS for 15 seconds
while (time.time() - start) < 15:
    # Get image from sim and find ORB keypoints
    rgb, depth = sim.step()
    featimg, kps, des = finder.get_orb(rgb)

    # Convert target and goal pixels to cartesian space
    target_pnt = ibvs.feature_to_pnt(target_pxl, depth)
    goal_pnt = ibvs.feature_to_pnt(goal_pxl, depth)

    #                           #
    #   Frame constellation     #
    #                           #
    if constellation is None:
        # Find translational distance from goal
        diff = goal_pnt - target_pnt

        # Assign feature closest to target pixel as root
        root_kp = finder.get_root(kps, target_pxl)
        kps.remove(root_kp)
        root_pxl = np.array(root_kp.pt)
        root_pnt = ibvs.feature_to_pnt(root_pxl, depth)

        # Assign n other features to be members of constellation
        feats = np.zeros((CONSTELLATION_SIZE, 2))
        constellation = np.zeros((CONSTELLATION_SIZE, 2))

        #                        #
        #   Feature selection    #
        #                        #
        for i in range(CONSTELLATION_SIZE):
            pxl = np.array(kps[i].pt)
            pnt = ibvs.feature_to_pnt(pxl, depth)
            feats[i] = pxl

            vec = pnt - root_pnt
            dist = np.linalg.norm(vec)
            angle = np.arctan(vec[1]/vec[0])
            constellation[i] = [dist, angle]

        # Set goal features and image jacobian
        ibvs.set_goal(feats, depth, diff)

    #                           #
    #   Match constellation     #
    #                           #
    else:
        min_sse = 0
        min_root = None
        feats = None
        for kp in kps:
            # Assign keypoint as root
            root_pxl = np.array(kp.pt)
            root_pnt = ibvs.feature_to_pnt(root_pxl, depth)
            rem_norms = list(kps)
            rem_norms.remove(kp)

            # Choose another keypoint to normalize constellation
            for kp in rem_norms:
                kp_pxl = np.array(kp.pt)
                kp_pnt = ibvs.feature_to_pnt(kp_pxl, depth)

                # Find difference in angle
                vec = kp_pnt - root_pnt
                angle = np.arctan(vec[1]/vec[0])
                d_angle = angle - constellation[0, 1]

                # Find set of features which minimize SSE from constellation
                sse = 0
                rem_kps = list(rem_norms)
                curr_feats = np.array([kp.pt])

                for pnt in constellation[1:]:
                    pnt = np.array(
                        [pnt[0] * np.cos(pnt[1] + d_angle), pnt[0] * np.sin(pnt[1] + d_angle), 0.0])
                    pnt += root_pnt
                    pxl = ibvs.pnt_to_feature(pnt)

                    # If error is greater than 35, don't bother checking
                    min_err = 250
                    min_err_kp = None
                    for kp in rem_kps:
                        kp_pxl = np.array(kp.pt)
                        kp_pnt = ibvs.feature_to_pnt(kp_pxl, depth)
                        err = np.linalg.norm(kp_pnt - pnt)
                        if err < min_err:
                            min_err = err
                            min_err_kp = kp

                    if min_err_kp is not None:
                        # Prevent feature from being reused in constellation
                        rem_kps.remove(min_err_kp)

                        # Attach feature pxl to current constellation build
                        curr_feats = np.vstack(
                            (curr_feats, min_err_kp.pt))
                    sse += (min_err**2)

                if min_root is None or sse < min_sse:
                    min_root = root_pxl
                    min_sse = sse
                    feats = curr_feats

    #
    #   Update
    #
    if constellation is not None:

        # Draw constellation to output image
        # for pxl in feats:
        #     pos = (int(pxl[0]), int(pxl[1]))
        #     cv.circle(featimg, pos, 5, (0, 255, 0), -1)

        print(feats)
        for pxl in feats:
            begin = ibvs.pnt_to_feature(root_pnt)
            end = (int(pxl[0]), int(pxl[1]))
            cv.line(featimg, tuple(begin.astype(int)),
                    end, (0, 255, 0), 5)

        # for pnt in constellation:
        #     begin = ibvs.pnt_to_feature(root_pnt)
        #     end = ibvs.pnt_to_feature(pnt + root_pnt)
        #     cv.line(featimg, tuple(begin.astype(int)),
        #             tuple(end.astype(int)), (0, 255, 0), 5)

        # Draw expected location of constellation to output image
        for pxl in ibvs._goal:
            pos = (int(pxl[0]), int(pxl[1]))
            cv.circle(featimg, pos, 5, (0, 125, 125), -1)

        if len(feats == 4):
            # Update velocity command
            vel = ibvs.execute(feats, depth)

            # Use velocity command to converge goal to target
            sim.applyVelocity(vel)

        if prev_root_pnt is not None:
            target_pnt += (root_pnt - prev_root_pnt)
        prev_root_pnt = root_pnt
    else:
        prev_root_pnt = None

    # Draw target
    target_pxl = ibvs.pnt_to_feature(target_pnt)
    cv.circle(featimg, tuple(target_pxl.astype(int)), 15, (0, 0, 255), -1)

    # Draw goal
    cv.circle(featimg, tuple(goal_pxl.astype(int)), 15, (125, 125, 0), -1)

    # Show image and sleep for sim
    cv.imshow("Output Image", featimg)
    cv.waitKey()
    time.sleep(sim.dt)

cv.destroyAllWindows()
sim.disconnect()
