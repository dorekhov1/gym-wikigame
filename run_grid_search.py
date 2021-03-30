import sys
import itertools

import gym
import torch
from tqdm import tqdm
from torch.multiprocessing import Pool, set_start_method
import graph_tool as gt

from config_parser import Configs
from agents.PPO import PPO
from agents.random_agent import *

try:
     set_start_method('spawn')
except RuntimeError:
    pass


class GridSearcher:
    
    def __init__(self):
        self.cf = Configs()
        if self.cf.random_seed:
            torch.manual_seed(self.cf.random_seed)
            np.random.seed(self.cf.random_seed)

        self.grid = {
            "optimize_timestep": [5, 10, 20],
            "k_epochs": [2, 5, 10],
            "eps_clip": [0.2],
            "gamma": [0.99],
            "lr": [0.02, 0.01, 0.005, 0.002, 0.001],
            "beta1": [0.9],
            "beta2": [0.999],
            "emb_dim": [16, 32, 64],
            "num_heads": [8, 16, 32, 64]
        }

    def run_experiment(self, agent_kwargs):

        env = gym.make(self.cf.env_name)
        env.seed(self.cf.random_seed)

        agent = getattr(sys.modules[__name__], self.cf.agent)(env.wiki_graph, agent_kwargs)

        running_reward = 0
        avg_length = 0
        avg_rewards = []

        for i_episode in range(1, self.cf.max_episodes + 1):
            state = env.reset()
            done = False
            while not done:
                action = agent.act(state)
                state, reward, done, _ = env.step(action)
                agent.update(reward, done)

                running_reward += reward

            avg_length += env.t

            if i_episode % self.cf.log_interval == 0:
                running_reward = (running_reward / self.cf.log_interval)
                avg_rewards.append(running_reward)

                running_reward = 0
                avg_length = 0

        return max(avg_rewards), agent_kwargs

    def generate_grid(self):
        keys, values = zip(*self.grid.items())
        permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
        return permutations_dicts

    def perform_grid_search(self):
        kwargs = self.generate_grid()
        with Pool(10) as p:
            results = list(tqdm(p.imap(self.run_experiment, kwargs), total=len(kwargs)))

        print("BEST RESULTS")
        results = sorted(results, key=lambda t: t[0])
        for r in results[-20:]:
            print(r)


if __name__ == "__main__":
    gs = GridSearcher()
    gs.perform_grid_search()
