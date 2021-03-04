import cv2 as cv
import numpy as np

class FeatureFinder(object):
    def __init__(self):
        self.sift = cv.xfeatures2d.SIFT_create()
        self.surf = cv.xfeatures2d.SURF_create(hessianThreshold=400)
        self.orb = cv.ORB_create()
    
    def get_sift(self, img):
        kp, des = self.sift.detectAndCompute(img, None)
        featimg = cv.drawKeypoints(img, kp, None, (255,0,0), 4)
        return featimg, kp, des

    def get_surf(self, img):
        kp, des = self.surf.detectAndCompute(img, None)
        featimg = cv.drawKeypoints(img, kp, None, (255,0,0), 4)
        return featimg, kp, des

    def get_orb(self, img):
        kp, des = self.orb.detectAndCompute(img, None)
        featimg = cv.drawKeypoints(img, kp, None, (255, 0, 0), 4)
        return featimg, kp, des