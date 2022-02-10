from twobin_environment import Environment
import numpy as np

env = Environment()
state = env.reset()
img = state[...,0:3].astype(np.uint8)

for i in range(10):
    H,W, _ = img.shape
    j = np.random.randint(H)
    i = np.random.randint(W)
    state, success = env.step([j,i])
    print("success:", success)
    img = state[...,0:3].astype(np.uint8)