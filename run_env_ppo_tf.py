from tensorforce.environments import Environment
from tensorforce.agents import Agent
from tensorforce.execution import Runner

import gym_wikigame
import gym_wikigame.envs
import gym_wikigame.envs.wikigame_env
from gym_wikigame.envs import WikigameEnv

environment = Environment.create(environment='gym', level='wikigame-v0', max_episode_timesteps=100)
agent = Agent.create(
    agent='ppo', environment=environment, batch_size=10, learning_rate=1e-3
)

runner = Runner(
    agent=agent,
    environment=environment
)
for _ in range(10):
    runner.run(num_episodes=20)
    runner.run(num_episodes=10, evaluation=True)

runner.close()
