from sim import Sim
from feature_finder import FeatureFinder
from feature_matcher import FeatureMatcher
import cv2 as cv
import numpy as np
import time
from ibvs import IBVS

sim = Sim(headless=True)
finder = FeatureFinder()
# For intel:
# ibvs = IBVS(np.array([800, 450]), 1280, 1204, 0.0000014)
ibvs = IBVS(np.array([225, 225]), 543.19, 543.19, 0.001)

start = time.time()
constellation = None
goal_pxl = np.array([230., 300.])
while (time.time() - start) < 60:
    # Get image from sim and find SURF keypoints
    rgb, depth = sim.step()
    featimg, kps, des = finder.get_surf(rgb)
    vel = None
    goal_pnt = ibvs.feature_to_pnt(goal_pxl, depth)

    if constellation is None:
        root = np.array(finder.get_root(kps, goal_pxl).pt)
        root_pnt = ibvs.feature_to_pnt(root, depth)
        diff = goal_pnt - root_pnt
        # print(diff)
        constellation_kps = np.zeros((4, 2))
        constellation = np.zeros((len(kps), 3))
        for i in range(4):
            constellation_kps[i] = np.array(kps[i].pt)
            kp = np.array(kps[i].pt)
            pnt = ibvs.feature_to_pnt(kp, depth)
            constellation[i] = pnt - root_pnt

        ibvs.set_goal(constellation_kps, depth, diff)
        vel = ibvs.execute(constellation_kps, depth)
    else:
        min_sse = 0
        min_root = None

        constellation_kps = None
        for root in kps:
            # root = finder.get_root(kps, goal_pnt)
            root_pnt = np.array(root.pt)
            root_pnt = ibvs.feature_to_pnt(root_pnt, depth)
            used_kps = []
            curr_constellation_kps = None
            sse = 0
            i = 0
            print(constellation)
            for pt in constellation:
                if i == 4:
                    break
                min_err = 35
                min_err_kp = None
                for kp in kps:
                    if kp not in used_kps:
                        kp_pnt = np.array(kp.pt)
                        kp_pnt = ibvs.feature_to_pnt(kp_pnt, depth)
                        if np.linalg.norm(kp_pnt - (root_pnt + pt)) < min_err:
                            min_err = np.linalg.norm(kp_pnt - (root_pnt + pt))
                            min_err_kp = kp
                if min_err_kp is not None:
                    used_kps.append(min_err_kp)
                    if curr_constellation_kps is None:
                        curr_constellation_kps = min_err_kp.pt
                    else:
                        curr_constellation_kps = np.vstack(
                            (curr_constellation_kps, min_err_kp.pt))
                sse += (min_err**2)
                i += 1
            print(sse)
            if min_root is None or sse < min_sse:
                min_root = root
                min_sse = sse
                constellation_kps = curr_constellation_kps

        if len(constellation_kps) < 4:
            constellation = None
        else:
            print(len(constellation_kps))
            root = np.array(min_root.pt)
            root_pnt = ibvs.feature_to_pnt(root, depth)

            if constellation is not None:
                for pt in constellation:
                    end = ibvs.pnt_to_feature(pt + root_pnt)
                    # print (end - root)
                    cv.line(featimg, tuple(root.astype(int)),
                            tuple(end.astype(int)), (0, 255, 0), 5)

            cv.line(featimg, tuple(root.astype(int)), tuple(
                goal_pxl.astype(int)), (255, 0, 0), 5)

            vel = ibvs.execute(constellation_kps, depth)

    if constellation is not None:
        # # Draw root
        # cv.circle(featimg, (root_x, root_y), 15, (255, 0, 0), -1
        for pxl in constellation_kps:
            pos = (int(pxl[0]), int(pxl[1]))
            # print(pos)
            cv.circle(featimg, pos, 5, (0, 255, 0), -1)

        # print(vel)
        vel[1] *= -1
        sim.applyVelocity(vel)

    for pxl in ibvs._goal:
        pos = (int(pxl[0]), int(pxl[1]))
        # print(pos)
        cv.circle(featimg, pos, 5, (0, 125, 125), -1)

    # Draw goal
    goal_pxl = ibvs.pnt_to_feature(goal_pnt)
    cv.circle(featimg, tuple(goal_pxl.astype(int)), 15, (0, 0, 255), -1)

    # Show image and sleep for sim
    cv.imshow("SIFT Features", featimg)
    cv.waitKey()
    time.sleep(sim.dt)

cv.destroyAllWindows()
sim.disconnect()
