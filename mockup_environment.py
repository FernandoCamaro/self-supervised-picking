import numpy as np
import cv2

class MockupEnvironment():
    def __init__(self):
        self.H, self.W = 256, 256
        self.num_blocks = 50
        self.block_size = 30
    
    def is_ok(self):
        return True

    def valid(self):
        valid_array = np.ones((self.H, self.W), dtype=np.bool)
        return valid_array.reshape(-1)
    
    def _new_state(self):
        rgb = np.zeros((self.H, self.W, 3))
        for i in range(self.num_blocks):
            j = np.random.randint(self.H)
            i = np.random.randint(self.W)
            jm = np.min([j+self.block_size, self.H])
            im = np.min([i+self.block_size, self.W])
            rgb[j:jm, i:im, :] = 1.
        noise = np.random.randn(16,16)
        depth = cv2.resize(noise, (self.H, self.W))

        self.rgb = rgb
        self.depth = depth

        return self.rgb.copy().transpose(2,0,1)#np.concatenate([rgb, depth[:,:,np.newaxis]], axis = 2)

    def max_reward(self):
        return 1.

    def reset(self):
        return self._new_state()

    def step(self, action):
        j = int(action / self.W)
        i = int(action % self.W) 
        if j>=0 and j<self.H and i>=0 and i<self.W:
            if self.rgb[j,i,0] == 1.: #and self.depth[j,i] > 0.:
                reward = 1.
            else:
                reward = 0.
        else:
            reward = 0.
        return self._new_state(), reward

    def sample_action(self):
        a = np.random.randint(self.H*self.W)
        return a

# import matplotlib.pyplot as plt

# env = MockupEnvironment()
# state = env.reset()
# # plt.imshow(state[:,:,0:3]); plt.show()
# # plt.imshow(state[:,:,3]); plt.show()
# _, H, W = state.shape
# rewards = []
# for i in range(100):
#     j = np.random.randint(H)
#     i = np.random.randint(W)
#     state, reward = env.step([j,i])
#     rewards.append(reward)

# print(np.mean(rewards))