from os import walk, path, mkdir
from feature_finder import FeatureFinder
from feature_matcher import FeatureMatcher
import cv2 as cv
import numpy as np

USE_SIFT = True
USE_SURF = True
USE_ORB = True
MATCH_FEATURES = True

finder = FeatureFinder()
matcher = FeatureMatcher()

for (root, dirs, files) in walk('raw_images'):
    save_path = root.split("/")

    if USE_SIFT:
        # SIFT path setup
        sift_path = save_path
        sift_path[0] = "sift_features"
        sift_path = "/".join(sift_path)
        if not path.isdir(sift_path):
            mkdir(sift_path)
    
    if USE_SURF:
        # SURF path setup
        surf_path = save_path
        surf_path[0] = "surf_features"
        surf_path = "/".join(surf_path)
        if not path.isdir(surf_path):
            mkdir(surf_path)

    if USE_ORB:
        # SURF path setup
        orb_path = save_path
        orb_path[0] = "orb_features"
        orb_path = "/".join(orb_path)
        if not path.isdir(orb_path):
            mkdir(orb_path)

        if MATCH_FEATURES:
            # BF feature match setup
            bf_orb_path = save_path
            bf_orb_path[0] = "orb_matches"
            bf_orb_path = "/".join(bf_orb_path)
            if not path.isdir(bf_orb_path):
                mkdir(bf_orb_path)

    has_prev = False
    prev_img = None
    prev_kp = None
    prev_des = None

    files.sort()
    for f in files:
        origimg = cv.imread("{}/{}".format(root, f), cv.IMREAD_GRAYSCALE)

        if USE_SIFT:
            # SIFT feature detection
            sift_file_path = "{}/{}".format(sift_path, f)
            featimg, kp, des = finder.get_sift(origimg)
            cv.imwrite(sift_file_path, featimg)

        if USE_SURF:
            # SURF feature detection
            surf_file_path = "{}/{}".format(surf_path, f)
            featimg, kp, des = finder.get_surf(origimg)
            cv.imwrite(surf_file_path, featimg)

        if USE_ORB:
            # ORB feature detection
            orb_file_path = "{}/{}".format(orb_path, f)
            featimg, kp, des = finder.get_orb(origimg)
            cv.imwrite(orb_file_path, featimg)

            if MATCH_FEATURES and len(kp) > 0:
                # ORB Feature matching
                bf_orb_file_path = "{}/{}".format(bf_orb_path, f)
                matchImg = matcher.get_orb(origimg, kp, des)
                cv.imwrite(bf_orb_file_path, matchImg)