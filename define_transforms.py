import numpy as np

T_zivid_to_world = np.eye(4)
T_zivid_to_world[0:3,0:3] = np.array([[0.0111658,  0.9582125, -0.2858393],
                                        [0.9985349,  0.0044512,  0.0539278],
                                        [0.0529466, -0.2860227, -0.9567589]])
T_zivid_to_world[0:3,3] = np.array([0.6586, -0.0573, 0.7017])
np.savetxt('transforms/T_zivid_to_world.txt', T_zivid_to_world)

T_world_to_robot = np.eye(4)
T_world_to_robot[0,0] = -1
T_world_to_robot[1,1] = -1
np.savetxt('transforms/T_world_to_robot.txt', T_world_to_robot)
