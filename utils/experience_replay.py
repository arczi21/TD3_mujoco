import numpy as np
import torch


class ExperienceReplay:
    def __init__(self, state_dim, action_dim, capacity, device):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.length = 0

        self.state = np.zeros((capacity, state_dim))
        self.action = np.zeros((capacity, action_dim))
        self.reward = np.zeros((capacity, 1))
        self.next_state = np.zeros((capacity, state_dim))
        self.is_done = np.zeros((capacity, 1))

    def __len__(self):
        return self.length

    def append(self, state, action, reward, next_state, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.is_done[self.ptr] = done

        self.length = min(self.length + 1, self.capacity)
        self.ptr = (self.ptr + 1) % self.capacity

    def sample(self, batch_size):
        idx = np.random.randint(0, self.length, batch_size)
        # idx = np.random.choice(range(0, self.length), batch_size, False)

        states = torch.FloatTensor(self.state[idx]).to(self.device)
        actions = torch.FloatTensor(self.action[idx]).to(self.device)
        rewards = torch.FloatTensor(self.reward[idx]).to(self.device)
        next_states = torch.FloatTensor(self.next_state[idx]).to(self.device)
        dones = torch.FloatTensor(self.is_done[idx]).to(self.device)

        return states, actions, rewards, next_states, dones
