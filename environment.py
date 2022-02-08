import numpy as np
import cv2
import time

from zivid_camera import ZividCamera
from robot import Robot

import pdb

class Environment():
    def __init__(self):
        robot_ip = "10.5.25.134"
        self.min_rob_z = -96e-3
        self.robot = Robot(robot_ip)
        self.camera = ZividCamera()

        T_zivid_to_world = np.eye(4)
        T_zivid_to_world[0:3,0:3] = np.array([[0.0111658,  0.9582125, -0.2858393],
                                              [0.9985349,  0.0044512,  0.0539278],
                                              [0.0529466, -0.2860227, -0.9567589]])
        T_zivid_to_world[0:3,3] = np.array([0.6586, -0.0573, 0.7017])
        T_world_to_robot = np.eye(4)
        T_world_to_robot[0,0] = -1
        T_world_to_robot[1,1] = -1
        self.T_zivid_robot = T_world_to_robot.dot(T_zivid_to_world)


    def reset(self):
        # TODO: move the robot to home position
        # TODO: I would like to handle two bins
        state = self._state()
        return state
    
    def _state(self):
        self.rgba, self.xyz, self.depth = self.camera.capture()
        state = np.concatenate([self.rgba[...,0:3], self.depth[:,:,np.newaxis]], axis = 2)
        return state

    def valid(self):
        return self.depth > 0.

    def step(self, action):
        """
        action is is the coordinates (j,i) of the pixel in which to perform the suction
        """
        j,i = action
        if self.depth[j,i] == -1:
            # The robot should be in the home position, but maybe I should check again
            state = self._state()
            reward = 0.
            return state, reward
        else:
            xyz_camera = self.xyz[j,i]*1e-3 # from mm to m.
            # command the robot to go the grasp point in a perpendicular direction
            xyz_camera_h = np.ones(4)
            xyz_camera_h[0:3] = xyz_camera
            xyz_robot_h = self.T_zivid_robot.dot(xyz_camera_h)
            xyz_robot = xyz_robot_h[0:3]
            xyz_robot[2] -= 0.05
            if xyz_robot[2] < self.min_rob_z:
                xyz_robot[2] = self.min_rob_z
            print("going to:", xyz_robot)
            T = np.eye(4)
            R = np.array([[0,1,0],[1.,0,0],[0,0,-1]])
            T[0:3,0:3] = R
            T[0:3, 3] = xyz_robot
            self.robot.go(T)
            self.robot.grasp()
            time.sleep(1)
            self.robot.go_home()
            self.robot.grasp(False)
            success = self.robot.heard_noise()
            time.sleep(1.0)
            state = self._state()
            return state, success


    def close(self):
        self.robot.close()


import matplotlib.pyplot as plt
from PIL import Image

def click_event(event, x, y, flags, params):
    global j,i
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        # print("click on:", y, x)
        j,i = y, x
        

env = Environment()
state = env.reset()
j,i = None, None
success = None
# # plt.imshow(state[...,0:3].astype(np.uint8)); plt.show()
for i in range(5):
    img = state[...,0:3].astype(np.uint8)

    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    state, success = env.step([j,i])
    print("success:", success)
    img = state[...,0:3].astype(np.uint8)