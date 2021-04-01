import pickle

import numpy as np
import gym
from graph_tool import topology
import torch
from torch.utils.data import Dataset

from config_parser import Configs

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class OptimalPolicyDataset(Dataset):

    def __init__(self, states, actions):

        masked_states = torch.nn.utils.rnn.pad_sequence(states, batch_first=False, padding_value=-1).to(device).detach()
        self.mask = masked_states == -1
        self.states = torch.nn.utils.rnn.pad_sequence(states, batch_first=False).to(device).detach()
        self.actions = torch.LongTensor(actions).to(device)

    def __len__(self):
        return len(self.states[0])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'state': self.states[:, idx], 'action': self.actions[idx], 'mask': self.mask[1:, idx]}

        return sample


class OptimalPolicyDataGenerator:

    def __init__(self):
        cf = Configs()
        self.env = gym.make(cf.env_name)
        self.states = []
        self.optimal_actions = []
        self.optimal_actions_one_hot = []
        self.dataset = None

    def generate_state_optimal_action_pair(self):
        state = self.env.reset()
        optimal_action = self.get_optimal_action(self.env.v_curr, self.env.v_goal)
        assert optimal_action in state
        self.states.append(torch.LongTensor(state))
        self.optimal_actions.append(np.where(state[1:] == optimal_action)[0][0])

    def get_optimal_action(self, v_curr, v_goal):
        paths = list(topology.all_shortest_paths(self.env.wiki_graph, v_curr, v_goal))
        return paths[0][1]

    def generate_dataset(self, n=100000):
        for _ in range(n):
            self.generate_state_optimal_action_pair()

        self.dataset = OptimalPolicyDataset(self.states, self.optimal_actions)
        return self.dataset

    def save_dataset(self):
        with open('data/optimal_policy_dataset.pickle', 'wb') as handle:
            pickle.dump(self.dataset, handle)


if __name__ == "__main__":
    g = OptimalPolicyDataGenerator()
    g.generate_dataset()
    g.save_dataset()
