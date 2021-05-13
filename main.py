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
constellation_pnts = None
target_pxl = np.array([230., 300.])
goal_pxl = np.array([225., 225.])
while (time.time() - start) < 10:
    # Get image from sim and find SURF keypoints
    rgb, depth = sim.step()
    featimg, kps, des = finder.get_surf(rgb)

    # Convert target and goal pixels to cartesian space
    target_pnt = ibvs.feature_to_pnt(target_pxl, depth)
    goal_pnt = ibvs.feature_to_pnt(goal_pxl, depth)

    if constellation_pnts is None:
        # Find distance from convergence
        diff = goal_pnt - target_pnt

        # Assign feature closest to target pixel as root
        root_kp = finder.get_root(kps, target_pxl)
        kps.remove(root_kp)
        root_pxl = np.array(root_kp.pt)
        root_pnt = ibvs.feature_to_pnt(root_pxl, depth)

        # Assign 4 other features to be members of constellation
        constellation_pxls = np.zeros((CONSTELLATION_SIZE, 2))
        constellation_pnts = np.zeros((CONSTELLATION_SIZE, 2))
        for i in range(CONSTELLATION_SIZE):
            pxl = np.array(kps[i].pt)
            pnt = ibvs.feature_to_pnt(pxl, depth)
            constellation_pxls[i] = pxl

            vec = pnt - root_pnt
            dist = np.linalg.norm(vec)
            angle = np.arctan(vec[1]/vec[0])
            constellation_pnts[i] = [dist, angle]

        # Set goal features and image jacobian
        ibvs.set_goal(constellation_pxls, depth, diff)

        # Get first velocity command
        vel = ibvs.execute(constellation_pxls, depth)
    else:
        min_sse = 0
        min_root = None
        constellation_pxls = None
        for kp in kps:
            # Assign keypoint as root
            root_pxl = np.array(kp.pt)
            root_pnt = ibvs.feature_to_pnt(root_pxl, depth)

            # Find set of features which minimize SSE from constellation
            sse = 0
            used_kps = []
            curr_constellation_pxls = None

            # Find kp with min euclidean distance for first constellation pnt
            min_err = 75
            min_err_kp = None
            for kp in kps:
                kp_pxl = np.array(kp.pt)
                kp_pnt = ibvs.feature_to_pnt(kp_pxl, depth)
                dist = np.linalg.norm(kp_pnt - root_pnt)
                err = dist - constellation_pnts[0, 0]
                if err < min_err:
                    min_err = err
                    min_err_kp = kp

            # Find difference in angle
            kp_pxl = np.array(min_err_kp.pt)
            kp_pnt = ibvs.feature_to_pnt(kp_pxl, depth)
            vec = kp_pnt - root_pnt
            print(kp_pnt)
            print(vec)
            angle = np.arctan(vec[1]/vec[0])
            d_angle = angle - constellation_pnts[0, 1]
            print(angle)
            # print(constellation_pnts[0, 1])

            for pnt in constellation_pnts[1:]:
                pnt = np.array(
                    [pnt[0] * np.cos(pnt[1] + d_angle), pnt[0] * np.sin(pnt[1] + d_angle), 0.0])
                pnt += root_pnt

                # If error is greater than 35, don't bother checking
                min_err = 75
                min_err_kp = None
                for kp in kps:
                    if kp not in used_kps:
                        kp_pxl = np.array(kp.pt)
                        kp_pnt = ibvs.feature_to_pnt(kp_pxl, depth)
                        err = np.linalg.norm(kp_pnt - pnt)
                        if err < min_err:
                            min_err = err
                            min_err_kp = kp

                if min_err_kp is not None:
                    # Prevent feature from being reused in constellation
                    used_kps.append(min_err_kp)

                    # Attach feature pxl to current constellation build
                    if curr_constellation_pxls is None:
                        curr_constellation_pxls = min_err_kp.pt
                    else:
                        curr_constellation_pxls = np.vstack(
                            (curr_constellation_pxls, min_err_kp.pt))
                sse += (min_err**2)

            if min_root is None or sse < min_sse:
                min_root = root_pxl
                min_sse = sse
                constellation_pxls = curr_constellation_pxls

        if len(constellation_pxls) < CONSTELLATION_SIZE:
            constellation_pnts = None
        else:
            root_pxl = min_root
            root_pnt = ibvs.feature_to_pnt(root_pxl, depth)

            # Draw line from target point to root
            cv.line(featimg, tuple(root_pxl.astype(int)), tuple(
                target_pxl.astype(int)), (255, 0, 0), 5)

            # Calculate iterative velocity command
            vel = ibvs.execute(constellation_pxls, depth)

    if constellation_pnts is not None:
        # Draw constellation to output image
        for pxl in constellation_pxls:
            pos = (int(pxl[0]), int(pxl[1]))
            cv.circle(featimg, pos, 5, (0, 255, 0), -1)

        # for pnt in constellation_pnts:
        #     begin = ibvs.pnt_to_feature(root_pnt)
        #     end = ibvs.pnt_to_feature(pnt + root_pnt)
        #     cv.line(featimg, tuple(begin.astype(int)),
        #             tuple(end.astype(int)), (0, 255, 0), 5)

        # Draw expected location of constellation to output image
        for pxl in ibvs._goal:
            pos = (int(pxl[0]), int(pxl[1]))
            cv.circle(featimg, pos, 5, (0, 125, 125), -1)

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
    cv.imshow("SIFT Features", featimg)
    cv.waitKey()
    time.sleep(sim.dt)

cv.destroyAllWindows()
sim.disconnect()
