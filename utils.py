import torch
from collections import namedtuple, deque
import random
from pathlib import Path
import os
import numpy as np
import pickle

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayBuffer(object):

    def __init__(self, capacity):
        self.memory_success = deque([],maxlen=capacity)
        self.memory_failure = deque([],maxlen=capacity)

    def push(self, state, action, next_state, reward):
        """Save a transition"""
        if reward == 1.0:
            memory = self.memory_success
        else:
            memory = self.memory_failure    

        st = torch.Tensor(state).unsqueeze(0)
        at = torch.LongTensor([action]).unsqueeze(0)
        nst = torch.Tensor(next_state).unsqueeze(0)
        rt = torch.Tensor([reward]).unsqueeze(0)
        memory.append(Transition(st, at, nst, rt))

    def sample(self, batch_size):
        positive_size = int(batch_size/2)
        negative_size = batch_size - positive_size
        pos = random.sample(self.memory_success, positive_size)
        neg = random.sample(self.memory_failure, negative_size)
        return pos+neg

    def __len__(self):
        len_success = len(self.memory_success)
        len_failure = len(self.memory_failure)
        return min(len_success, len_failure)

class ReplayBufferDisk(object):

    def __init__(self, capacity, folder):
        self.capacity = capacity
        self.folder = Path(folder)
        os.makedirs(self.folder, exist_ok=True)
        self.success_folder = self.folder.joinpath('success')
        self.failure_folder = self.folder.joinpath('failure')
        self.num_success = 0
        self.num_failure = 0

    def push(self, state, action, next_state, reward):
        """Save a transition"""
        if reward == 1.0:
            folder = self.success_folder
            sub_folder_num = self.num_success
            self.num_success += 1

        else:
            folder = self.failure_folder
            sub_folder_num = self.num_failure
            self.num_failure += 1

        subfolder = folder.joinpath(sub_folder_num)
        os.makedirs(subfolder)
        np.save(state, subfolder.joinpath('state'))
        np.save(next_state, subfolder.joinpath('next_state'))
        with open(subfolder.joinpath('action_reward.pkl'),'wb') as f:
            pickle.dump([action, reward], f)

    def load_transition(self, folder):
        state = np.load(folder.joinpath('state.npy'))
        next_state = np.load(folder.joinpath('next_state.npy'))

    def sample(self, batch_size):
        positive_size = int(batch_size/2)
        negative_size = batch_size - positive_size
        samples = []
        for i in range(positive_size):
            idx = np.random.randint(self.num_success)
            folder = self.success_folder.joinpath(idx)
            sample = self.load_transition(folder)
            samples.append()
        pos = random.sample(self.memory_success, positive_size)
        neg = random.sample(self.memory_failure, negative_size)
        return pos+neg

    def __len__(self):
        len_success = len(self.memory_success)
        len_failure = len(self.memory_failure)
        return min(len_success, len_failure)

class EpsilonScheduler():
    def __init__(self, initial_epsilon, final_epsilon, num_steps):
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.num_steps = num_steps
        self.step = 0
    
    def update(self):
        theta = self.step / self.num_steps
        eps = theta*self.final_epsilon + (1-theta)*self.initial_epsilon
        self.step += 1
        return eps

class Bunch(object):
  def __init__(self, adict):
    self.__dict__.update(adict)

opt = {}
opt['display_id'] = 1
opt['isTrain'] = True 
opt['no_html'] = False 
opt['name'] = "ssp" 
opt['display_winsize'] = 256 
opt['display_port'] = 8097 
opt['display_ncols'] = 4 
opt['display_freq'] = 1 
opt['display_server'] = "http://localhost" 
opt['display_env'] = "main" 
opt['checkpoints_dir'] = "./checkpoints" 
opt['update_html_freq'] = "100"


opt = Bunch(opt)
