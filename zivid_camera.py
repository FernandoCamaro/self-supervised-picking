import zivid

import matplotlib.pyplot as plt
import pptk
import numpy as np

class ZividCamera():
    def __init__(self):
        app = zivid.Application()
        self.camera = app.connect_camera()
        self.settings = zivid.Settings(acquisitions=[zivid.Settings.Acquisition()])

    def capture(self):
        frame = self.camera.capture(self.settings)
        point_cloud = frame.point_cloud()
        xyz = point_cloud.copy_data("xyz")
        depth = xyz[:,:,2]
        invalid = np.isnan(depth)
        depth[invalid] = -1
        rgba = point_cloud.copy_data("rgba")
        return rgba, xyz, depth