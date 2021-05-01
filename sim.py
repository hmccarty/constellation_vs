import pybullet as p
import pybullet_data
import cv2 as cv
import numpy as np
import transforms3d as tf3d

class Sim(object):
    def __init__(self, headless=False):
        if headless:
            self._client = p.connect(p.DIRECT)
        else:
            self._client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, 0)
        self.dt = 1./240.
        
        # Load and texture plane with features
        planeId = p.loadURDF('plane.urdf')
        textureId = p.loadTexture('untitled.png')
        p.changeVisualShape(planeId, -1, textureUniqueId=textureId)

        cubeId = p.loadURDF('cube.urdf', [0, 0, 1], p.getQuaternionFromEuler([0,0,0]), globalScaling=1.5)
        textureId = p.loadTexture('texture.png')
        p.changeVisualShape(cubeId, -1, textureUniqueId=textureId)
        
        # Setup camera params
        self.width = 450
        self.height = 450
        self.fov = 45.0
        self.aspect = 1
        self.nearVal = 0.1
        self.farVal = 5.0
        self._projMatrix = p.computeProjectionMatrixFOV(fov=self.fov,
                                                        aspect=self.aspect,
                                                        nearVal=self.nearVal,
                                                        farVal=self.farVal)
        
        self.camPosition = np.array([0., 0., 4.])
        self.targetPosition = np.array([0., 0., 0.])
        self.upVector = np.array([0., 1., 0.])

    def step(self):
        p.stepSimulation()
        viewMatrix = p.computeViewMatrix(self.camPosition,
                                         self.targetPosition,
                                         self.upVector)

        # Capture and return single image frame
        _, _, rgb, depth, _ = p.getCameraImage(width=self.width,
                                            height=self.height,
                                            viewMatrix=viewMatrix,
                                            projectionMatrix=self._projMatrix,
                                            renderer=p.ER_BULLET_HARDWARE_OPENGL,
                                            flags=p.ER_NO_SEGMENTATION_MASK)
        return cv.cvtColor(rgb, cv.COLOR_RGBA2RGB), depth

    def applyVelocity(self, tf):
        # Translate camera
        translation = tf[:3] * self.dt
        translation[1] *= -1
        self.camPosition += translation
        self.targetPosition += translation

        # Rotate camera
        rotation = tf[3:] * self.dt
        rotation = tf3d.euler.euler2mat(rotation[0], rotation[1], rotation[2])
        unit = self.targetPosition - self.camPosition
        self.targetPosition = self.camPosition + np.matmul(rotation, unit)
        self.upVector = np.matmul(rotation, self.upVector)
    
    def getTargetDist(self):
        return np.linalg.norm(self.targetPosition - self.camPosition)

    def disconnect(self):
        p.disconnect()