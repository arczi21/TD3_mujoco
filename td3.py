import torch
import torch.nn as nn
import torch.optim as optim
from utils.actor import Actor
from utils.critic import Critic


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def update_parameters(target_net, net, tau):
    for target_param, param in zip(target_net.parameters(), net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


class TD3:
    def __init__(self, state_dim, action_dim, action_abs, actor_lr=0.0003, critic_lr=0.0003, gamma=0.99, tau=0.005,
                 policy_noise_std=0.2, policy_noise_clip=0.5, batch_size=256, policy_update_freq=2):
        self.policy_noise_std = policy_noise_std * action_abs
        self.policy_noise_clip = policy_noise_clip * action_abs
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_abs = action_abs
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.policy_update_freq = policy_update_freq

        self.actor = Actor(state_dim, action_dim, action_abs).to(device)
        self.actor_target = Actor(state_dim, action_dim, action_abs).to(device)
        update_parameters(self.actor_target, self.actor, 1.0)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        update_parameters(self.critic_target, self.critic, 1.0)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.step = 1
        self.criterion = nn.MSELoss()

    def action(self, state):
        with torch.no_grad():
            state_t = torch.FloatTensor([state]).to(device)
            action = self.actor(state_t)[0]
            action = action.cpu().numpy()
        return action

    def update_model(self, batch):
        state, action, reward, next_state, done = batch

        # critic
        noise = torch.randn((self.batch_size, self.action_dim)).to(device) * self.policy_noise_std
        noise = noise.clamp(-self.policy_noise_clip, self.policy_noise_clip)

        with torch.no_grad():
            next_action = self.actor_target(next_state) + noise
            next_action.clamp(-self.action_abs, self.action_abs)

            target1, target2 = self.critic_target(next_state, next_action)
            target = torch.min(target1, target2)
            target = reward + (1 - done) * self.gamma * target

        out1, out2 = self.critic(state, action)
        loss = self.criterion(out1, target) + self.criterion(out2, target)

        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        # actor
        if self.step % self.policy_update_freq == 0:
            loss = - self.critic.q1(state, self.actor(state)).mean()

            self.actor_optimizer.zero_grad()
            loss.backward()
            self.actor_optimizer.step()

            update_parameters(self.actor_target, self.actor, self.tau)
            update_parameters(self.critic_target, self.critic, self.tau)

        self.step += 1


