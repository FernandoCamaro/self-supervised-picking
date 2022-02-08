import pptk
import numpy as np
import open3d as o3d
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--p", help="point cloud (ply)", required=True)
parser.add_argument("--s", help="scale to apply to the point cloud", type = float, required=True)
args = parser.parse_args()

pcd = o3d.io.read_point_cloud(args.p)

def pick_points(pcd):
    print("")
    print(
        "1) Please pick at least three correspondences using [shift + left click]"
    )
    print("   Press [shift + right click] to undo point picking")
    print("2) Afther picking points, press q for close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()

points = pick_points(pcd)
points = np.array(pcd.points)[points]
output_path = args.p.replace('.ply', '_points.txt')
np.savetxt(output_path, points*args.s)