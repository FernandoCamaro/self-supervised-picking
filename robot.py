import urx
import numpy as np
import time

class Robot():
    def __init__(self, robot_ip = "10.5.51.54"):
        self._rob = urx.Robot(robot_ip)
        self._home = self._rob.get_pos()

    def grasp(self, point=None):
        pos = self._home - np.array([0,0,0.05])
        self._rob.set_pos(pos, wait=True)
        self._rob.set_digital_out(0,1)
        time.sleep(1)
        self._rob.set_pos(self._home, wait=True)
        self._rob.set_digital_out(0,0)
        return self._rob.get_digital_out_bits()

    def close(self):
        self._rob.close()

rob = Robot()

