import argparse

import gym
import torch
import parser
from hyperparameters import *
from td3 import TD3
from utils.experience_replay import ExperienceReplay
from utils.noise import GaussianNoise


def evaluate(policy, t_env, n_episodes=10):
    reward_sum = 0
    for episode in range(n_episodes):
        state = t_env.reset()
        done = False
        while not done:
            action = policy.action(state)
            state, reward, done, _ = t_env.step(action)
            reward_sum += reward
    return reward_sum / n_episodes


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env", default="Hopper-v2")
    parser.add_argument("--actor_lr", default=ACTOR_LR)
    parser.add_argument("--critic_lr", default=CRITIC_LR)
    parser.add_argument("-t", "--tau", default=TAU)
    parser.add_argument("-g", "--gamma", default=GAMMA)
    parser.add_argument("--exploration_std", default=EXPLORATION_NOISE_STD)
    parser.add_argument("--policy_noise_std", default=POLICY_NOISE_STD)
    parser.add_argument("--policy_noise_clip", default=POLICY_NOISE_CLIP)
    parser.add_argument("-d", "--policy_update_freq", default=POLICY_UPDATE_FREQ)
    parser.add_argument("--max_steps", default=MAX_STEPS)
    parser.add_argument("-r", "--replay_capacity", default=REPLAY_CAPACITY)
    parser.add_argument("-b", "--batch_size", default=BATCH_SIZE)
    parser.add_argument("--min_replay_size", default=MIN_REPLAY_SIZE)
    parser.add_argument("--eval_freq", default=EVAL_FREQ)
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_env = gym.make(args.env)
    test_env = gym.make(args.env)
    state_dim = train_env.observation_space.shape[0]
    action_dim = train_env.action_space.shape[0]
    action_abs = train_env.action_space.high[0]

    print(f"env={args.env}, state_dim={state_dim}, action_dim={action_dim}, action_abs={action_abs}")

    td3 = TD3(state_dim, action_dim, action_abs, actor_lr=args.actor_lr, critic_lr=args.critic_lr, gamma=args.gamma,
              tau=args.tau, policy_noise_std=args.policy_noise_std, policy_noise_clip=args.policy_noise_clip,
              batch_size=args.batch_size, policy_update_freq=args.policy_update_freq)
    experience_replay = ExperienceReplay(state_dim, action_dim, REPLAY_CAPACITY, device)
    noise = GaussianNoise(action_dim, 0, args.exploration_std * action_abs)

    state = train_env.reset()

    for step in range(args.max_steps):
        if len(experience_replay) < args.min_replay_size:
            action = train_env.action_space.sample()
        else:
            action = td3.action(state) + noise.noise()
        next_state, reward, done, _ = train_env.step(action)
        experience_replay.append(state, action, reward, next_state, done)
        state = next_state
        if done:
            state = train_env.reset()
        if len(experience_replay) >= args.min_replay_size:
            batch = experience_replay.sample(BATCH_SIZE)
            td3.update_model(batch)
        if (step + 1) % EVAL_FREQ == 0:
            mean_reward = evaluate(td3, test_env)
            print('Step %d | Mean reward: %.3f' % (step + 1, mean_reward))
            torch.save(td3.actor, 'actor.pkl')


