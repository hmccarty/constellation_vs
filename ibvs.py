import cv2 as cv
import numpy as np

class IBVS(object):
    def __init__(self, principal_pnt, focal_x, focal_y, pxl_size):
        self._principal_pnt = principal_pnt
        self._focal_x = focal_x
        self._focal_y = focal_y
        self._pxl_size = pxl_size
        self._lambda = 0.5
        self._L = None
        self._goal = None
        self._Z = 0.0
        self._constellation = None

    def set_goal(self, features, depth, diff):
        '''
        Sets goal features from goal pnt and current features.

            Parameters:
                features (kx2 numpy array): features found in image
        '''

        for feature in features:
            if self._L is None:
                self._L = self._get_jacobian(feature, depth, diff)
            else:
                self._L = np.vstack((self._L, self._get_jacobian(feature, depth, diff)))
        self._goal = np.array(features)

    def execute(self, features, depth):
        '''
        Determines velocity command to reach goal features from given features

            Parameters:
                features (kx2 numpy array): features found in image

            Returns:
                command (6x1 numpy array): 6DoF velocity command
        '''
        err = features - self._goal
        err = err.flatten()
        vel = self._lambda * -np.dot(np.linalg.pinv(self._L), err)
        return vel

    def _get_jacobian(self, feature, depth, diff):
        '''
        Derives jacobian matrix for a given feature point

            Parameters:
                feature (1x3 numpy array): pos of a single feature
                goal (1x2 numpy goal): desired pos of feature
            
            Returns:
                jacobian (2x6 numpy array): the iteraction matrix for a feature
        '''
        pnt = self.feature_to_pnt(feature, depth)
        pnt += diff
        x = pnt[0]
        y = pnt[1]
        Z = pnt[2]
        return np.array([[-1/Z, 0, x/Z, x*y, -(1+x*x), y],
                         [0, -1/Z, y/Z, 1+y*y, -x*y, -x]])

    def feature_set_to_pnts(self, features, depth):
        pnts = np.array([self.feature_to_pnt(np.array(f.pt), depth) for f in features])
        return pnts

    def kps_to_feature(self, kps):
        feature = np.zeros((len(kps), 2))
        for i in range(len(kps)):
            feature[i, 0] = int(kps[i].pt[0])
            feature[i, 1] = int(kps[i].pt[1])
        return feature

    def feature_to_pnt(self, feature, depth):
        '''
        Converts given feature into a point using camera features

            Parameters:
                feature (1x2 numpy array): pos of a single feature
                depth (wxh numpy array): physical depth of every pixel
            
            Returns:
                pnt (1x3 numpy array): pos of the cartesian pnt
        '''
        pnt = feature - self._principal_pnt
        pnt[0] /= (self._focal_x * self._pxl_size)
        pnt[1] /= (self._focal_y * self._pxl_size)
        z = depth[int(feature[1]), int(feature[0])]
        pnt *= z
        return np.append(pnt, z)

    def pnt_to_feature(self, pnt):
        feature = pnt[:2]
        if pnt[2] != 0:
            feature /= pnt[2]
        feature[0] *= (self._focal_x * self._pxl_size)
        feature[1] *= (self._focal_y * self._pxl_size)
        feature += self._principal_pnt
        return feature