import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_abs):
        super(Actor, self).__init__()
        self.action_abs = action_abs

        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)

    def forward(self, state):
        p = F.relu(self.fc1(state))
        p = F.relu(self.fc2(p))
        p = self.action_abs * torch.tanh(self.fc3(p))
        return p
