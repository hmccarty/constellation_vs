import pybullet as p
import pybullet_data
import cv2 as cv
import numpy as np

class Sim(object):
    def __init__(self, headless=False):
        if headless:
            self._client = p.connect(p.DIRECT)
        else:
            self._client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        self.dt = 1./240.
        
        # Load and texture plane with features
        planeId = p.loadURDF('plane.urdf')
        textureId = p.loadTexture('texture.png')
        p.changeVisualShape(planeId, -1, textureUniqueId=textureId)
        
        # Setup camera params
        self.width = 224
        self.height = 224
        self.fov = 45.0
        self.aspect = 1.0
        self.nearVal = 0.1
        self.farVal = 3.1
        self._projMatrix = p.computeProjectionMatrixFOV(fov=self.fov,
                                                        aspect=self.aspect,
                                                        nearVal=self.nearVal,
                                                        farVal=self.farVal)
        
        self.camPosition = np.array([0., 0., 3.])
        self.targetPosition = np.array([0., 0., 0.])
        self.upVector = np.array([0., 1., 0.])

    def step(self):
        p.stepSimulation()
        viewMatrix = p.computeViewMatrix(self.camPosition,
                                         self.targetPosition,
                                         self.upVector)

        # Capture and return single image frame
        _, _, img, _, _ = p.getCameraImage(width=self.width,
                                            height=self.height,
                                            viewMatrix=viewMatrix,
                                            projectionMatrix=self._projMatrix)
        return cv.cvtColor(img, cv.COLOR_RGBA2RGB)

    def applyVelocity(self, cmd):
        self.camPosition += (cmd * self.dt)
    
    def getTargetDist(self):
        return np.linalg.norm(self.targetPosition - self.camPosition)

    def disconnect(self):
        p.disconnect()