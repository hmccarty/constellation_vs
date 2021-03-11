import cv2 as cv
import numpy as np

class IBVS(object):
    def __init__(self, principal_pnt, focal_len, pixel_dim):
        self.principal_pnt = principal_pnt
        self.focal_len = focal_len
        self.pixel_dim = pixel_dim
        self.lambda = 0.5
        self.goal = None

    def set_goal(self, goal, features):
        '''
        Sets goal features from goal pnt and current features.

            Parameters:
                goal (3x1 numpy array): point selected to be tracked
                features (3xk numpy array): features found in image
        '''
        pass
    
    def execute(self, features):
        '''
        Determines velocity command to reach goal features from given features

            Parameters:
                features (3xk numpy array): features found in image

            Returns:
                command (6x1 numpy array): 6DoF velocity command
        '''
        pass

    def _get_jacobian(self, feature):
        '''
        Derives jacobian matrix for a given feature point

            Parameters:
                feature (3x1 numpy array): a single feature
            
            Returns:
                jacobian (2x6 numpy array): the iteraction matrix for a feature
        '''
        jacobian = 
        pass