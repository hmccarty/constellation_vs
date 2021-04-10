from sim import Sim
from feature_finder import FeatureFinder
from feature_matcher import FeatureMatcher
import cv2 as cv
import numpy as np
import time

sim = Sim()
finder = FeatureFinder()

start = time.time()
while (time.time() - start) < 20:
    if sim.getTargetDist() > 1.5:
        sim.applyVelocity(np.array([0., 0., -0.75]))
    img = sim.step()
    featimg, kp, des = finder.get_sift(img)
    cv.imshow("SIFT Features", featimg)
    cv.waitKey()
    time.sleep(sim.dt)
sim.disconnect()