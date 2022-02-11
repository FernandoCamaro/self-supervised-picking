from ast import Num
import torch
import numpy as np

from mockup_environment import MockupEnvironment
from models import Model
from dqn_traininer import DQNTrainer
from logger import Logger
from utils import ReplayBuffer, EpsilonScheduler, opt
from visualizer import Visualizer

# hyper-parameters
num_steps = 10000
initial_epsilon = 0.9
final_epsilon = 0.15
gamma = 0.5
buffer_capacity = 5000

visualizer = Visualizer(opt)
env = MockupEnvironment()
max_reward = env.max_reward()
state = env.reset()
H, W, _ = state.shape
device = torch.device("cuda:0")
model = Model(gamma, max_reward)
model.to(device)
replay_buffer = ReplayBuffer(buffer_capacity)
trainer = DQNTrainer(model, replay_buffer, gamma)
eps_sch = EpsilonScheduler(initial_epsilon, final_epsilon, num_steps)

for step in range(num_steps):
    if env.is_ok():
        
        # select action
        eps = eps_sch.update()
        random = np.random.rand() < eps
        if random:
            a = env.sample_action()
        else:
            q = model(state) # [H,W]
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
        visualizer.plot_current_losses(0, step / num_steps, losses)
        # logger.report(step, random, transition)
        
        

