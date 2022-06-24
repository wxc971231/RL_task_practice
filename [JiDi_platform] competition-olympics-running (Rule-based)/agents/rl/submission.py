import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import numpy as np


device = 'cpu'

class Actor(nn.Module):
    def __init__(self, state_space, action_space, hidden_size=64, cnn=False):
        super(Actor, self).__init__()
        self.is_cnn = cnn
        self.linear_in = nn.Linear(state_space, hidden_size)
        self.action_head = nn.Linear(hidden_size, action_space)

    def forward(self, x):
        if self.is_cnn:
            x = self.encoder(x)
        x = F.relu(self.linear_in(x))
        action_prob = F.softmax(self.action_head(x), dim=1)
        return action_prob


class Critic(nn.Module):
    def __init__(self, state_space, hidden_size=64, cnn=False):
        super(Critic, self).__init__()
        self.is_cnn = cnn
        self.linear_in = nn.Linear(state_space, hidden_size)
        self.state_value = nn.Linear(hidden_size, 1)

    def forward(self, x):
        if self.is_cnn:
            x = self.encoder(x)
        x = F.relu(self.linear_in(x))
        value = self.state_value(x)
        return value

actions_map = {0: [-100, -30], 1: [-100, -18], 2: [-100, -6], 3: [-100, 6], 4: [-100, 18], 5: [-100, 30], 6: [-40, -30],
               7: [-40, -18], 8: [-40, -6], 9: [-40, 6], 10: [-40, 18], 11: [-40, 30], 12: [20, -30], 13: [20, -18],
               14: [20, -6], 15: [20, 6], 16: [20, 18], 17: [20, 30], 18: [80, -30], 19: [80, -18], 20: [80, -6],
               21: [80, 6], 22: [80, 18], 23: [80, 30], 24: [140, -30], 25: [140, -18], 26: [140, -6], 27: [140, 6],
               28: [140, 18], 29: [140, 30], 30: [200, -30], 31: [200, -18], 32: [200, -6], 33: [200, 6], 34: [200, 18],
               35: [200, 30]}           #dicretise action space

class RLAgent(object):
    def __init__(self, obs_dim, act_dim, num_agent):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_agent = num_agent
        self.device = 'cpu'
        self.actor = Actor(self.obs_dim, self.act_dim).to(self.device)

    def choose_action(self, obs):
        state = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_prob = self.actor(state).to(self.device)

        action = torch.argmax(action_prob)
        return action.item()

    def load_model(self, filename):
        self.actor.load_state_dict(torch.load(filename))

agent = RLAgent(25*25, 36, 1)
actor_net = os.path.dirname(os.path.abspath(__file__)) + "/actor_1500.pth"
agent.load_model(actor_net)


def my_controller(observation_list, action_space_list, is_act_continuous):
    obs_dim = 25*25
    obs = observation_list['obs'].copy().flatten()
    actions_raw = agent.choose_action(obs)
    actions = actions_map[actions_raw]
    wrapped_actions = [[actions[0]], [actions[1]]]
    return wrapped_actions

