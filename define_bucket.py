import numpy as np
import pickle

Twz = np.loadtxt("transforms/T_zivid_to_world.txt")
Trw = np.loadtxt("transforms/T_world_to_robot.txt")
Trz= Trw.dot(Twz)

zivid_points = np.loadtxt('pointclouds/bucket_points.txt')
points = np.ones((3,4))
points[:,0:3] = zivid_points[0:3,:]
points = points.T

origin = Trz.dot(points[:,0])[0:3]
p1 = Trz.dot(points[:,1])[0:3]
p2 = Trz.dot(points[:,2])[0:3]
z_axis = np.array([0,0,1.])
x_axis = p2 - origin
norm_x = np.linalg.norm(x_axis)
x_axis = x_axis/norm_x
y_axis = np.cross(z_axis, x_axis)
y_axis = y_axis/np.linalg.norm(y_axis)
norm_y = np.linalg.norm(p1 - origin)
T_rb = np.eye(4)
T_rb[0:3,0] = x_axis
T_rb[0:3,1] = y_axis
T_rb[0:3,2] = z_axis
T_rb[0:3,3] = origin

bucket = {'T_rb':T_rb, 'width':norm_x, 'length':norm_y}
with open('transforms/bucket.pkl','bw') as f:
    pickle.dump(bucket, f)