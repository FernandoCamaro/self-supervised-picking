import urx
import numpy as np
import time
import math3d
import sounddevice as sd
from scipy.spatial.transform import Rotation 

import socket

class Robot():
    def __init__(self, robot_ip = "10.5.51.54"):
        self._rob = urx.Robot(robot_ip, use_rt=False)
        self._home = self._rob.get_pos()
        self.robot_ip = robot_ip
        self.tcp_port = 30002

    def grasp(self, on=True):
        robot_ip = "10.5.25.134"
        tcp_port = 30002
        tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tcp_socket.connect((robot_ip, tcp_port))
        if on==True:
            tcp_command = "set_analog_out(0,0.3)\n"
        else:
            tcp_command = "set_analog_out(0,0.0)\n"
        tcp_socket.send(str.encode(tcp_command))
        tcp_socket.close()

    def get_force(self):
        values = []
        for i in range(100):
            v = self._rob.get_force()
            values.append(v)
        return np.mean(values)



    def close(self):
        self._rob.close()

    def go(self, T, vel=0.5, acc=0.3):
        self._rob.set_pose(T2math3d(T), vel=vel, acc=acc, wait=True)

    def go_home(self, vel=0.5, acc=0.3):
        self._rob.set_pos(self._home, vel=vel, acc=acc, wait=True)

    def suction(self, on=True):
        if on:
            self._rob.set_digital_out(8,0)
            self._rob.set_digital_out(9,1)
        else:
            self._rob.set_digital_out(8,1)
            self._rob.set_digital_out(9,0)

    def heard_noise(self):
        fs=44100
        duration = 2
        self.recording = sd.rec(duration * fs, samplerate=fs, channels=1, dtype='float64')
        sd.wait()
        self.recording = self.recording[20000:]
        if np.sum(self.recording[:,0] > 0.3) > 10:
            return True
        else:
            return False 
        

    def suction_success(self):
        return self._rob.get_digital_in(17)

    def go_grasp_and_retrieve(self, T, vel=0.5, acc=0.3):
        cmd = grasp_and_retrieve_cmd(T, self._home, vel, acc)
        tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tcp_socket.connect((self.robot_ip, self.tcp_port))
        tcp_socket.send(str.encode(cmd))
        tcp_socket.close()


def T2math3d(T):
    trans = math3d.Transform()
    trans.orient = T[0:3,0:3]
    trans.pos = T[0:3, 3]
    return trans

def grasp_and_retrieve_cmd(T, home, vel, acc):
    p_grasp = [T[0,3], T[1,3], T[2,3]]
    o_grasp = Rotation.from_matrix(T[0:3,0:3]).as_rotvec()

    cmd_lines = [
        "def process():",
        " global graspoint_p=p[%f,%f,%f,%f,%f,%f]" % (p_grasp[0], p_grasp[1], p_grasp[2], o_grasp[0], o_grasp[1], o_grasp[2]),
        " global home_p=p[%f,%f,%f,%f,%f,%f]" % (home[0], home[1], home[2], o_grasp[0], o_grasp[1], o_grasp[2]),
        " global move_thread_flag_7=0",
        " thread move_thread_7():",
        "   enter_critical",
        "   move_thread_flag_7 = 1",
        "   movel(graspoint_p, a=1.2, v=0.25)",
        "   move_thread_flag_7 = 2",
        "   exit_critical",
        " end",
        " move_thread_flag_7 = 0",
        " move_thread_han_7 = run move_thread_7()",
        " while (True):",
        "   if ( force ()>20):",
        "     kill move_thread_han_7",
        "     stopl(1.2)",
        "     break",
        "   end",
        "   sleep(1.0E-10)",
        "   if (move_thread_flag_7 > 1):",
        "     join move_thread_han_7",
        "     break",
        "   end",
        "   sync()",
        " end",
        " set_analog_out(0,0.3)",
        " sleep(1.)",
        " movel(home_p, a=1.2, v=0.25)",
        "end"]
    cmd = ""
    for x in cmd_lines:
        cmd += x+'\n'
    return cmd

# rob = Robot("10.5.25.134") # UR 1
# rob = Robot("10.5.51.54") # UR 2