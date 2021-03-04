import cv2 as cv
import numpy as np

class FeatureMatcher(object):
    IMG = 0
    KP = 1
    DES = 2

    FLANN_INDEX_KDTREE = 1
    INDEX_PARAMS = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    SEARCH_PARAMS = dict(checks=50)

    def __init__(self):
        # Setup ORB for brute-force matching
        self.orb = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        self.orbprev = None

        # Setup SIFT for Flann matching
        self.sift = cv.FlannBasedMatcher(self.INDEX_PARAMS, self.SEARCH_PARAMS)
        self.siftprev = None

    def get_orb(self, img, kp, des):
        matchImg = None

        # Check if the given frame has trackable features
        if len(kp) > 0:
            # Check if the previous frame had trackable features
            if self.orbprev != None:
                # Perform brute-force matching then sort by distance
                matches = self.orb.match(self.orbprev[self.DES], des)
                matches = sorted(matches, key = lambda x:x.distance)

                # Draw matches on new image
                matchImg = cv.drawMatches(self.orbprev[self.IMG], self.orbprev[self.KP],
                                          img, kp,
                                          matches[:10], None,
                                          flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            self.orbprev = (img, kp, des)
        else:
            self.orbprev = None
        return matchImg