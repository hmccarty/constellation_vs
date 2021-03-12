from ibvs import IBVS
import numpy as np
import cv2 as cv
from feature_finder import FeatureFinder
from feature_matcher import FeatureMatcher

def size(k):
    return k.size

finder = FeatureFinder()
matcher = FeatureMatcher()

img1 = cv.imread("2.jpg", cv.IMREAD_GRAYSCALE)
img2 = cv.imread("4.jpg", cv.IMREAD_GRAYSCALE)

featimg, kp1, des = finder.get_orb(img1)
matcher.get_orb(img1, kp1, des)
cv.imwrite("1-feat.jpg", finder.draw_keypoints(img1, kp1))

featimg, kp2, des = finder.get_orb(img2)
cv.imwrite("2-feat.jpg", finder.draw_keypoints(img2, kp2))
match = matcher.get_orb(img2, kp2, des)
cv.imwrite("match.jpg", match)

matches = matcher.matches
feature = None
goal = None
for match in matches:
    x1, y1 = kp1[match.trainIdx].pt
    x2, y2 = kp2[match.queryIdx].pt

    if feature is not None:
        feature = np.vstack((feature, np.array([x1, y1])))
    else:
        feature = np.array([x1, y2])
    if goal is not None:
        goal = np.vstack((goal, np.array([x2, y2])))
    else:
        goal = np.array([x2, y2])

i = IBVS(np.array([[4032/2,3024/2]]), 27.2224, 1.333)
i.set_goal(np.array(goal), 1)
print(i.execute(feature))