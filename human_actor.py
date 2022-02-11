from twobin_environment import Environment
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

def click_event(event, x, y, flags, params):
    global j,i
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        # print("click on:", y, x)
        j,i = y, x
        

env = Environment()
state = env.reset()
j,i = 0, 0
reward = None
# # plt.imshow(state[...,0:3].astype(np.uint8)); plt.show()
for i in range(10):
    img = (state.transpose(1,2,0)*255).astype(np.uint8)
    im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    cv2.imshow('image', im_rgb)
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    state, reward = env.step([j,i])
    print("rewards:", reward)
    img = state[...,0:3].astype(np.uint8)