import numpy as np
import cv2
import time
from PIL import Image

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

        self.right_coord = [1055,251]
        self.left_coord = [83,251]
        self.right_home = np.array([-462, -312, 161])*1e-3
        self.left_home = np.array([-462, 259, 161])*1e-3
        self.crop_size = 800
        self.H, self.W = 256, 256

        self.current_bin = "right"
        self.counter = 1

        valid_img = Image.open('misc/valid_new.png')
        self.valid_img = np.array(valid_img)[:,:,0]

    def is_ok(self):
        return True

    def valid_mask(self):
        valid_array = np.ones((self.H, self.W), dtype=np.bool)
        return valid_array.reshape(-1)

    def max_reward(self):
        return 1.

    def reset(self):
        # TODO: move the robot to home position
        # TODO: I would like to handle two bins
        state = self._state()
        return state
    
    def _state(self):
        self.rgba, self.xyz, self.depth = self.camera.capture()
        self._particularize_for_current_bin()
        # state = np.concatenate([self.rgba[...,0:3], self.depth[:,:,np.newaxis]], axis = 2)
        state = self.rgba[...,0:3].transpose(2,0,1)/255.
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
        self.rgba = self.rgba[v:v+self.crop_size,u:u+self.crop_size]
        self.xyz = self.xyz[v:v+self.crop_size,u:u+self.crop_size]
        self.depth = self.depth[v:v+self.crop_size,u:u+self.crop_size]
        self.valid = self.valid_img[v:v+self.crop_size,u:u+self.crop_size]
        self.valid = (self.valid == 255) * (self.depth > 0.7)

        self.rgba = cv2.resize(self.rgba, [self.W, self.H])
        self.xyz = cv2.resize(self.xyz, [self.W, self.H], interpolation=cv2.INTER_NEAREST)
        self.depth = cv2.resize(self.depth, [self.W, self.H], interpolation=cv2.INTER_NEAREST)
        valid = self.valid*1.0
        valid = cv2.resize(valid, [self.W, self.H], interpolation=cv2.INTER_NEAREST)
        self.valid = valid == 1.0

    def step(self, action):
        """
        action is is the coordinates (j,i) of the pixel in which to perform the suction
        """
        j = int(action / self.W)
        i = int(action % self.W) 
        if ~self.valid[j,i]:
            # The robot should be in the home position, but maybe I should check again
            state = self._state()
            reward = False
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
            T = np.eye(4)
            R = np.array([[0,1,0],[1.,0,0],[0,0,-1]])
            T[0:3,0:3] = R
            T[0:3, 3] = xyz_robot
            self.robot.go_grasp_and_retrieve(T, self.bin_home, self.drop)
            time.sleep(7)
            self.robot.grasp(False)
            reward = 1.0 if self.robot.heard_noise() else 0.
            self.robot.move_home(T)
            time.sleep(2.0)
            state = self._state()
            self.counter += 1
            if self.counter == 6:
                self.counter = 1
                self.current_bin = "right" if self.current_bin == "left" else "left"
            return state, reward

    def sample_action(self):
        a = np.random.randint(self.H*self.W)
        return a

    def close(self):
        self.robot.close()