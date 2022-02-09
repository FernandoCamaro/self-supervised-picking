import socket
import pickle
import sounddevice as sd
import numpy as np
from scipy.spatial.transform import Rotation

import pdb

class Robot():
    def __init__(self, robot_ip = "10.5.51.54"):
        with open("transforms/home.pkl",'rb') as f:
            self._home = pickle.load(f).pos
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

    def _send_command(self, cmd):
        tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tcp_socket.connect((self.robot_ip, self.tcp_port))
        tcp_socket.send(str.encode(cmd))
        tcp_socket.close()

    def go_grasp_and_retrieve(self, T, bin, drop, vel=0.5, acc=0.3):
        
        p_grasp = [T[0,3], T[1,3], T[2,3]]
        o_grasp = Rotation.from_matrix(T[0:3,0:3]).as_rotvec()

        cmd_lines = [
            "def process():",
            " global graspoint_p=p[%f,%f,%f,%f,%f,%f]" % (p_grasp[0], p_grasp[1], p_grasp[2], o_grasp[0], o_grasp[1], o_grasp[2]),
            " global bin_p=p[%f,%f,%f,%f,%f,%f]" % (bin[0], bin[1], bin[2], o_grasp[0], o_grasp[1], o_grasp[2]),
            " global drop_p=p[%f,%f,%f,%f,%f,%f]" % (drop[0], drop[1], drop[2], o_grasp[0], o_grasp[1], o_grasp[2]),
            " global move_thread_flag_7=0",
            " thread move_thread_7():",
            "   enter_critical",
            "   move_thread_flag_7 = 1",
            "   movel(graspoint_p, a=1.2, v=0.25)",
            "   move_thread_flag_7 = 2",
            "   exit_critical",
            " end",
            " movel(bin_p, a=1.2, v=0.25, r=0.05)",
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
            " movel(bin_p, a=1.2, v=0.25, r=0.05)",
            " movel(drop_p, a=1.2, v=0.25)",
            "end"]
        cmd = self._prepare_cmd(cmd_lines)
        self._send_command(cmd)

    def move_home(self, T):
        o_grasp = Rotation.from_matrix(T[0:3,0:3]).as_rotvec()
        home = self._home
        cmd_lines = [
            "def process():",
            " global home_p=p[%f,%f,%f,%f,%f,%f]" % (home[0], home[1], home[2], o_grasp[0], o_grasp[1], o_grasp[2]),
            " movel(home_p, a=1.2, v=0.25)",
            "end"]
        cmd = self._prepare_cmd(cmd_lines)
        self._send_command(cmd)

    def _prepare_cmd(self, cmd_lines):
        cmd = ""
        for x in cmd_lines:
            cmd += x+'\n'
        return cmd