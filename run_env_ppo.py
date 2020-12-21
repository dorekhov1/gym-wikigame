import gym
import graph_tool as gt
import torch

from config_parser import Configs
from agents.PPO_continuous import PPO, Memory
from agents.simple_agents import *


def main():

    cf = Configs('PPO')

    # creating environment
    env = gym.make(cf.env_name)
    state_dim = 512
    action_dim = 512

    wiki_graph = gt.load_graph(cf.graph_path)
    prop_embeddings = wiki_graph.vertex_properties["embedding"]

    env.set_graph(wiki_graph)

    if cf.random_seed:
        torch.manual_seed(cf.random_seed)
        env.seed(cf.random_seed)
        np.random.seed(cf.random_seed)

    memory = Memory()
    ppo = PPO(state_dim, action_dim, cf.action_std, cf.lr, cf.betas, cf.gamma, cf.k_epochs, cf.eps_clip)

    # logging variables
    running_reward = 0
    avg_length = 0
    time_step = 0

    # training loop
    for i_episode in range(1, cf.max_episodes + 1):
        state = env.reset()
        goal_state_embedding = prop_embeddings[state[1]]
        for t in range(cf.max_timesteps):
            time_step += 1

            start_state_embedding = prop_embeddings[state[0]]

            state_embedding = np.concatenate((start_state_embedding, goal_state_embedding))

            # Running policy_old:
            action = ppo.select_action(goal_state_embedding, memory)
            state, reward, done, _ = env.step(action)

            # Saving reward and is_terminals:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            # update if its time
            if time_step % cf.update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                time_step = 0
            running_reward += reward
            if cf.render:
                env.render()
            if done:
                # print(f"Done in {t} steps")
                break

        avg_length += t

        # stop training if avg_reward > solved_reward
        if running_reward > (cf.log_interval * cf.solved_reward):
            print("########## Solved! ##########")
            torch.save(ppo.policy.state_dict(), './PPO_continuous_solved_{}.pth'.format(cf.env_name))
            break

        # save every 500 episodes
        if i_episode % 500 == 0:
            torch.save(ppo.policy.state_dict(), './PPO_continuous_{}.pth'.format(cf.env_name))

        # logging
        if i_episode % cf.log_interval == 0:
            avg_length = int(avg_length / cf.log_interval)
            running_reward = int((running_reward / cf.log_interval))

            print('Episode {} \t Avg length: {} \t Avg reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0


if __name__ == "__main__":
    main()
