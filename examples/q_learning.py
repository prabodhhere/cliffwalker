import sys

if "../" not in sys.path:
  sys.path.append("../")

from lib.envs.cliff_walking import CliffWalkingEnv
from lib import plotting
from agents import QLearningAgent
import numpy as np

env_shape = (4, 12)
start_position = (3, 0)
end_positions = [(3, 11)]
cliff = tuple((3, i+1) for i in range(10))

env = CliffWalkingEnv(env_shape, start_position, end_positions, cliff)
n_actions = env.action_space.n
agent = QLearningAgent(alpha=0.5, epsilon=0.1, discount=0.99, n_actions=n_actions)

agent.train(env, n_episodes=5000, t_max=10**3, verbose=True, verbose_per_episode=500)

plotting.draw_policy(env, agent)
