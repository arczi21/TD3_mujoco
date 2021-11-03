import gym
import torch
import time
import numpy as np
from hyperparameters import *
from utils.experience_replay import ExperienceReplay
from utils.critic import Critic
from utils.actor import Actor
from td3 import TD3

device = torch.device('cuda:0')
env_name = 'Ant-v2'
actor = torch.load('actor.pkl')
env = gym.make(env_name)

while True:
    state = env.reset()
    done = False
    cr = 0
    while not done:
        state_t = torch.FloatTensor([state]).to(device)
        action = actor(state_t)[0]
        action = action.detach().cpu().numpy()
        env.render()
        state, reward, done, _ = env.step(action)
        cr += reward
    print(cr)