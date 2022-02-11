import torch
from collections import namedtuple, deque
import random

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
