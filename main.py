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
goal_pnt = np.array([230, 300, 0.5])
while (time.time() - start) < 60:
    # Move towards plane
    if sim.getTargetDist() > 1.5:
        sim.applyVelocity(np.array([0., 0., -1.5, 0., 0., 0.]))

    # Get image from sim and find SURF keypoints
    rgb, depth = sim.step()
    featimg, kps, des = finder.get_surf(rgb)

    if constellation is None:
        root = np.array(finder.get_root(kps, goal_pnt).pt)
        root_pnt = ibvs._feature_to_pnt(root, depth)
        diff = goal_pnt - root_pnt
        constellation = np.zeros((len(kps), 3))
        for i in range(len(kps)):
            kp = np.array(kps[i].pt)
            pnt = ibvs._feature_to_pnt(kp, depth)
            constellation[i] = pnt - root_pnt
        print (len(kps))
        ibvs.set_goal(kps, depth, diff)
        print(ibvs.execute(kps, depth))
    else:
        max = 0
        maxRoot = None
        for root in kps:
            root = finder.get_root(kps, goal_pnt)
            cnt = 0
            root_pnt = np.array(root.pt)
            root_pnt = ibvs._feature_to_pnt(root_pnt, depth)

            for pt in constellation:
                for kp in kps:
                    kp_pnt = np.array(kp.pt)
                    kp_pnt = ibvs._feature_to_pnt(kp_pnt, depth)
                    if np.linalg.norm(kp_pnt - (root_pnt + pt)) < 10:
                        cnt += 1
                        break

            if maxRoot is None or cnt > max:
                max = cnt
                maxRoot = root
            break

        root = np.array(maxRoot.pt)
        root_pnt = ibvs._feature_to_pnt(root, depth)

        for pt in constellation:
            end = ibvs.pnt_to_pxl(pt + root_pnt)
            print (end - root)
            cv.line(featimg, tuple(root.astype(int)), tuple(end.astype(int)), (0, 255, 0), 5)

    # Find keypoint closest to goal point
    # For each keypoint 
        # Check each constellation point
        # Count number of matches
        # If number of matches is == to size of constellation
            # Use constellation
        # Else check if max
            # Set max
    # Use max matches
    # Draw constellation
    # Update goal by constellation root
    # root = finder.get_root(kp, goal_pnt)
    # root_x = int(root.pt[0])
    # root_y = int(root.pt[1])

    # # Draw root
    # cv.circle(featimg, (root_x, root_y), 15, (255, 0, 0), -1)

    # Draw goal
    cv.circle(featimg, tuple(goal_pnt), 15, (0, 0, 255), -1)

    # Show image and sleep for sim
    cv.imshow("SIFT Features", featimg)
    cv.waitKey()
    # time.sleep(sim.dt)
sim.disconnect()