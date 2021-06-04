from os import walk, path, mkdir
import cv2 as cv
import numpy as np


class FeatureFinder(object):
    def __init__(self):
        # self.sift = cv.xfeatures2d.SIFT_create()
        # self.surf = cv.xfeatures2d.SURF_create(hessianThreshold=400)
        self.orb = cv.ORB_create()

    def get_sift(self, img):
        # kp, des = self.sift.detectAndCompute(img, None)
        # kp = kp[-10:]
        # featimg = cv.drawKeypoints(img, kp, None, (255, 0, 0), 4)
        # return featimg, kp, des
        pass

    def get_surf(self, img):
        # kp, des = self.surf.detectAndCompute(img, None)
        # kp = kp[:6]
        # featimg = cv.drawKeypoints(img, kp, None, (255, 0, 0), 4)
        # return featimg, kp, des
        pass

    def get_orb(self, img):
        kp, des = self.orb.detectAndCompute(img, None)
        # kp = kp[:30]
        kp = kp[:5]
        featimg = cv.drawKeypoints(img, kp, None, (255, 0, 0), 4)
        return featimg, kp, des

    def get_root(self, kps, goal_pxl):
        min = np.linalg.norm(goal_pxl - np.array(kps[0].pt))
        min_kp = kps[0]
        for kp in kps[1:]:
            dist = np.linalg.norm(goal_pxl - np.array(kp.pt))
            if dist < min:
                min = dist
                min_kp = kp
        return min_kp

    def get_closest_idxs(self, kps, goal_pxl):
        dists = np.array()
        for kp in kps:
            dists.append(np.linalg.norm(goal_pxl - np.array(kp.pt)))
        return np.argsort(dists)

    def draw_keypoints(self, img, kp):
        return cv.drawKeypoints(img, kp, None, (255, 0, 0), 4)


USE_SIFT = False
USE_SURF = False
USE_ORB = True

if __name__ == "__main__":
    finder = FeatureFinder()

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
