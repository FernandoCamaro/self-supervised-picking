import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import copy
from collections import OrderedDict


from utils import Transition


BATCH_SIZE = 8
TARGET_UPDATE = 250
LR = 1e-3

class DQNTrainer():
    def __init__(self, policy_net, replay_buffer, gamma):

        self.gamma = gamma

        # models
        self.policy_net = policy_net
        self.target_net = copy.deepcopy(policy_net)
        self.target_net.eval()
        

        # optimizer
        self.optimizer = optim.RMSprop(policy_net.parameters(), lr=LR)

        # experience replay
        self.buffer = replay_buffer

        self.criterion = nn.SmoothL1Loss()

        self.num_updates = 0
        self.loss = 0.
        self.td_error = 0.

    def optimize_model(self):
        if len(self.buffer) < BATCH_SIZE:
            return
        else:
            device = self.policy_net.device
            transitions = self.buffer.sample(BATCH_SIZE)
            # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
            # detailed explanation). This converts batch-array of Transitions
            # to Transition of batch-arrays.
            batch = Transition(*zip(*transitions))

            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)
            next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])

            # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
            # columns of actions taken. These are the actions which would've been taken
            # for each batch state according to policy_net
            state_action_values = self.policy_net(state_batch).gather(1, action_batch.to(device))

            # Compute V(s_{t+1}) for all next states.
            # Expected values of actions for non_final_next_states are computed based
            # on the "older" target_net; selecting their best reward with max(1)[0].
            # This is merged based on the mask, such that we'll have either the expected
            # state value or 0 in case the state was final.
            next_state_values = self.target_net(next_states).max(1)[0].detach()
            # Compute the expected Q values
            expected_state_action_values = (next_state_values * self.gamma) + reward_batch.to(device)

            # Compute Huber loss
            # self.td_error = (state_action_values - expected_state_action_values.unsqueeze(1)).abs().mean().item()
            loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))
            self.loss = loss.item()
            # print("loss:", loss.item())

            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            for param in self.policy_net.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()

            # Update the target network, copying all weights and biases in DQN
            if self.num_updates % TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            self.num_updates += 1

    def return_losses_for_visualizer(self):
        od = OrderedDict()
        od['Huber(TD_error)'] = self.loss
        od['TD_error'] = self.td_error
        return od