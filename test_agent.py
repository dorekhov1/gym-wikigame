import gym
import torch
import numpy as np
from graph_tool.topology import shortest_distance

from config_parser import Configs
from agents.PPO import ActionValueLayer


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():

    cf = Configs()
    env = gym.make(cf.env_name)

    if cf.random_seed:
        torch.manual_seed(cf.random_seed)
        env.seed(cf.random_seed)
        np.random.seed(cf.random_seed)

    agent = torch.load("models/action_layer.pth").to(device)

    num_episodes = 1000

    goal_reached_counter = 0
    shortest_path_counter = 0

    for i_episode in range(1, num_episodes + 1):
        state = env.reset()
        done = False

        shortest_dist = shortest_distance(
            env.wiki_graph, env.v_curr, env.v_goal, directed=True
        )

        episode = {
            "start": env.p_titles[env.v_curr],
            "goal": env.p_titles[env.v_goal],
            "path": [env.p_titles[env.v_curr]],
            "shortest_path_length": shortest_dist
        }

        while not done:
            prob_action = agent.act(torch.LongTensor(state).to(device))
            action = prob_action.argmax()
            state, reward, done, _ = env.step(action)
            episode["path"].append(env.p_titles[env.v_curr])

        episode['path_length'] = len(episode['path']) - 1

        if episode['goal'] == episode['path'][-1]:
            goal_reached_counter += 1
            if episode['path_length'] == episode['shortest_path_length']:
                shortest_path_counter += 1

    print(f'Percentage of goal reached: {goal_reached_counter / num_episodes}')
    print(f'Percentage of shortest path: {shortest_path_counter / num_episodes}')


if __name__ == "__main__":
    main()
