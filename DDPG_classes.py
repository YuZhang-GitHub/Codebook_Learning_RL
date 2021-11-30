import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Critic(nn.Module):

    def __init__(self, input_size, output_size):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(input_size, 16 * input_size)
        self.fc2 = nn.Linear(16 * input_size, 16 * output_size)
        self.fc3 = nn.Linear(16 * output_size, output_size)

    def forward(self, state, action):
        x = torch.cat((state, action), 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class Actor(nn.Module):

    def __init__(self, input_size, output_size):
        super(Actor, self).__init__()
        self.pi = torch.tensor(np.pi).float().cuda()

        self.fc1 = nn.Linear(input_size, 16 * input_size)
        self.fc2 = nn.Linear(16 * input_size, 16 * output_size)
        self.fc3 = nn.Linear(16 * output_size, output_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x)) * self.pi

        return x


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.uniform_(m.weight, -0.01, 0.01)
        m.bias.data.fill_(0.01)


class OUNoise(object):
    def __init__(self, action_shape, mu=0.0, theta=0.15, max_sigma=1, min_sigma=0.05, decay_period=100000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_shape
        self.low = -np.pi
        self.high = np.pi
        self.state = self.reset()

    def reset(self):
        state = torch.ones(self.action_dim) * self.mu
        return state.float().cuda()

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * torch.normal(0, 1, size=self.action_dim).cuda()
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        # return torch.clamp(action + ou_state, self.low, self.high)
        return action + ou_state
