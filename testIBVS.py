from ibvs import IBVS
import numpy as np
import cv2 as cv
from feature_finder import FeatureFinder
from feature_matcher import FeatureMatcher


def size(k):
    return k.size


finder = FeatureFinder()
matcher = FeatureMatcher()

img1 = cv.imread("103.jpg", cv.IMREAD_GRAYSCALE)
img2 = cv.imread("104.jpg", cv.IMREAD_GRAYSCALE)


def test(img1, img2):
    featimg, kp1, des = finder.get_orb(img1)
    matcher.get_orb(img1, kp1, des)
    cv.imwrite("1-feat.jpg", finder.draw_keypoints(img1, kp1))

    featimg, kp2, des = finder.get_orb(img2)
    cv.imwrite("2-feat.jpg", finder.draw_keypoints(img2, kp2))
    match = matcher.get_orb(img2, kp2, des)
    cv.imwrite("match.jpg", match)


test(img1, img2)
