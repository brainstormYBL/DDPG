import torch
import torch.nn as nn
import torch.nn.functional as f


class Actor(nn.Module):
    def __init__(self, dim_state, dim_action):
        super(Actor, self).__init__()
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.fc1 = nn.Linear(self.dim_state, 256)
        self.fc2 = nn.Linear(256, 256)
        self.out = nn.Linear(256, dim_action)

    def forward(self, x):
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        actions = torch.tanh(self.out(x))

        return actions


class Critic(nn.Module):
    def __init__(self, dim_state, dim_action):
        super(Critic, self).__init__()
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.fc1 = nn.Linear(self.dim_state + self.dim_action, 256)
        self.fc2 = nn.Linear(256, 256)
        self.q_out = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        q_value = self.q_out(x)
        return q_value
