import torch
import numpy as np
from collections import OrderedDict

from mockup_environment import MockupEnvironment
from models import Model
from dqn_traininer import DQNTrainer
from logger import Logger
from utils import ReplayBuffer, EpsilonScheduler, opt
from visualizer import Visualizer

import matplotlib.pyplot as plt

# hyper-parameters
num_steps = 10000
initial_epsilon = 0.6
final_epsilon = 0.15
gamma = 0.5
buffer_capacity = 5000

visualizer = Visualizer(opt)
env = MockupEnvironment()
max_reward = env.max_reward()
state = env.reset()
_, H, W = state.shape
device = torch.device("cuda:0")
model = Model(gamma, max_reward)
model.to(device)
replay_buffer = ReplayBuffer(buffer_capacity)
trainer = DQNTrainer(model, replay_buffer, gamma)
eps_sch = EpsilonScheduler(initial_epsilon, final_epsilon, num_steps)
q = None
for step in range(num_steps):
    if env.is_ok():
        
        # select action
        eps = eps_sch.update()
        random = np.random.rand() < eps
        if random:
            a = env.sample_action()
        else:
            q = model(state) # [H,W]
            q_s = q.copy()
            state_vis = state.copy()
            valid = env.valid()
            q[~valid] = -np.inf
            a = np.argmax(q)
            
        # environment step
        new_state, reward = env.step(a)
        
        # accumulate experience
        replay_buffer.push(state, a, new_state, reward)

        # update and log
        trainer.optimize_model()
        visualizer.reset()
        losses = trainer.return_losses_for_visualizer()
        visualizer.plot_current_losses(step, losses)
        if step % opt.display_freq == 0 and q is not None:
            
            visuals = OrderedDict()
            q_vis = q_s.reshape(H,W)/model.max_possible_qvalue*255.
            q_vis = np.stack([q_vis, q_vis, q_vis])
            # visuals['action_value'] = q_vis
            # visualizer.display_current_results(visuals, 0, False)
            
            # visuals['state'] = state_vis*255.
            # visualizer.display_current_results(visuals, 0, False)
            images = [q_vis, state_vis*255.]
            visualizer.vis.images(images, nrow=2, win=2, opts=dict(title='images'))
            

        state = new_state
        
        

