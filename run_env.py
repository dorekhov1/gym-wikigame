import sys

import gym
import torch

from config_parser import Configs
from agents.PPO import PPO
from agents.random_agent import *


def main():

    cf = Configs()
    env = gym.make(cf.env_name)

    if cf.random_seed:
        torch.manual_seed(cf.random_seed)
        env.seed(cf.random_seed)
        np.random.seed(cf.random_seed)

    agent = getattr(sys.modules[__name__], cf.agent)(env.wiki_graph, cf.get_agent_kwargs())

    running_reward = 0
    avg_length = 0
    avg_rewards = []

    for i_episode in range(1, cf.max_episodes + 1):
        state = env.reset()
        done = False
        while not done:
            action = agent.act(state)
            state, reward, done, _ = env.step(action)
            agent.update(reward, done)

            running_reward += reward

        avg_length += env.t

        if running_reward > (cf.log_interval * cf.solved_reward):
            print("########## Solved! ##########")
            agent.save(True)
            break

        if i_episode % cf.save_interval == 0:
            agent.save()

        if i_episode % cf.log_interval == 0:
            avg_length = int(avg_length / cf.log_interval)
            running_reward = (running_reward / cf.log_interval)
            avg_rewards.append(running_reward)

            print('Episode {} \t Avg length: {} \t Avg reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0


if __name__ == "__main__":
    main()
