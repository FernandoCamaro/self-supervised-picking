import numpy as np
import cv2
import time
import pickle

from zivid_camera import ZividCamera
from robot import Robot

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

        with open('transforms/bucket.pkl','rb') as f:
            self.bucket = pickle.load(f)
            self.Trb = self.bucket["T_rb"]
            self.Tbc = (np.linalg.inv(self.Trb)).dot(self.T_zivid_robot)


    def reset(self):
        # TODO: move the robot to home position
        # TODO: I would like to handle two bins
        state = self._state()
        return state
    
    def _state(self):
        self.rgba, self.xyz, self.depth = self.camera.capture()
        state = np.concatenate([self.rgba[...,0:3], self.depth[:,:,np.newaxis]], axis = 2)
        self._ortographic_state()
        return state

    def _ortographic_state(self):
        valid = self.depth > 0
        xyz = self.xyz[valid]*1e-3 # -1, 3
        rgba = self.rgba[valid]
        N, _ = xyz.shape
        xyz_hc = np.concatenate([xyz,np.ones((N, 1))], axis=1).T # 4, -1
        xyz_hb = self.Tbc.dot(xyz_hc) # homogeneous points in bucket frame
        width = self.bucket['width']
        pixels = 256
        self.height_img = np.ones((pixels, pixels))*-0.01
        self.color_img = np.zeros((pixels, pixels, 3), dtype=np.uint8)
        self.xyz_img = np.zeros((pixels, pixels, 3))
        for i, p in enumerate(xyz_hb.T):
            x, y, z = p[0], p[1], p[2]
            v = np.int(x*(pixels-1)/width)
            u = np.int(y*(pixels-1)/width)
            if u>0 and u<pixels and v>0 and v<pixels:
                if z > self.height_img[v,u]:
                    self.height_img[v,u] = z
                    self.color_img[v,u] = rgba[i,0:3]
                    self.xyz_img[v,u] = xyz[i]

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
            self.robot.go_grasp_and_retrieve(T)
            time.sleep(3)
            while(self.robot._rob.is_program_running()):
                time.sleep(0.01)
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
    im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    cv2.imshow('image', im_rgb)
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    state, success = env.step([j,i])
    print("success:", success)
    img = state[...,0:3].astype(np.uint8)