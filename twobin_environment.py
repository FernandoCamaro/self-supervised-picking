import numpy as np
import cv2
import time
import pickle

from zivid_camera import ZividCamera
from robot_no_urx import Robot

import pdb

class Environment():
    def __init__(self):
        robot_ip = "10.5.25.134"
        self.min_rob_z = -96e-3
        self.robot = Robot(robot_ip)
        self.camera = ZividCamera()

        Twz = np.loadtxt("transforms/T_zivid_to_world.txt")
        Trw = np.loadtxt("transforms/T_world_to_robot.txt")
        self.T_zivid_robot = Trw.dot(Twz)

        self.right_coord = [1085,447]
        self.left_coord = [255,447]
        self.right_home = np.array([-468, -159, 178])*1e-3
        self.left_home = np.array([-468, 191, 178])*1e-3

        self.current_bin = "right"
        self.counter = 1

    def reset(self):
        # TODO: move the robot to home position
        # TODO: I would like to handle two bins
        state = self._state()
        return state
    
    def _state(self):
        self.rgba, self.xyz, self.depth = self.camera.capture()
        self._particularize_for_current_bin()
        state = np.concatenate([self.rgba[...,0:3], self.depth[:,:,np.newaxis]], axis = 2)
        return state

    def _particularize_for_current_bin(self):
        if self.current_bin == "right":
            u,v = self.right_coord
            self.bin_home = self.right_home
            self.drop = self.left_home
        else:
            u,v = self.left_coord
            self.bin_home = self.left_home
            self.drop = self.right_home
        self.rgba = self.rgba[v:v+512,u:u+512]
        self.xyz = self.xyz[v:v+512,u:u+512]
        self.depth = self.depth[v:v+512,u:u+512]

    def valid(self):
        return self.depth > 0.

    def step(self, action):
        """
        action is is the coordinates (j,i) of the pixel in which to perform the suction
        """
        j,i = action
        if self.depth[j,i] < 1e-2:
            # The robot should be in the home position, but maybe I should check again
            state = self._state()
            reward = 0.
            return state, reward
        else:
            xyz_camera = self.xyz[j,i]*1e-3
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
            self.robot.go_grasp_and_retrieve(T, self.bin_home, self.drop)
            time.sleep(7)
            self.robot.grasp(False)
            success = self.robot.heard_noise()
            print(success)
            self.robot.move_home(T)
            time.sleep(2.0)
            state = self._state()
            self.counter += 1
            if self.counter == 2:
                self.counter = 1
                self.current_bin = "right" if self.current_bin == "left" else "left"
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
j,i = 0, 0
success = None
# # plt.imshow(state[...,0:3].astype(np.uint8)); plt.show()
for i in range(10):
    img = state[...,0:3].astype(np.uint8)
    im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    cv2.imshow('image', im_rgb)
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    state, success = env.step([j,i])
    print("success:", success)
    img = state[...,0:3].astype(np.uint8)