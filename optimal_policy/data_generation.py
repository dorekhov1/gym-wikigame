import pickle

import gym
from graph_tool import topology

from config_parser import Configs


class OptimalPolicyDataGenerator:

    def __init__(self):
        cf = Configs()
        self.env = gym.make(cf.env_name)
        self.states = []
        self.optimal_actions = []

    def generate_state_optimal_action_pair(self):
        state = self.env.reset()
        optimal_action = self.get_optimal_action(self.env.v_curr, self.env.v_goal)
        assert optimal_action in state
        self.states.append(state)
        self.optimal_actions.append(optimal_action)

    def get_optimal_action(self, v_curr, v_goal):
        paths = list(topology.all_shortest_paths(self.env.wiki_graph, v_curr, v_goal))
        return paths[0][1]

    def generate_dataset(self, n=10000):
        for _ in range(n):
            self.generate_state_optimal_action_pair()

    def save_dataset(self):
        d = {
            'states': self.states,
            'actions': self.optimal_actions
        }
        with open('data/optimal_policy_dataset.pickle', 'wb') as handle:
            pickle.dump(d, handle)


if __name__ == "__main__":
    g = OptimalPolicyDataGenerator()
    g.generate_dataset()
    g.save_dataset()
