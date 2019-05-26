# -*- coding: utf-8 -*-
"""
Created on Fri May 17 21:03:14 2019

@author: shane
"""

import sys
import gym
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

# Set plotting options
#%matplotlib inline
plt.style.use('ggplot')
np.set_printoptions(precision=3, linewidth=120)


"""
TODO

1. generalize different agents to handle different environments to compare them
2. try the idea of sleeping the network with a different memory buffer size
3. try transfer learning between environments (batch norm to help?)
4. try to generalize agents between different environments

"""

"""
# Create an environment and set random seed
"""

# Toy Text - Discrete state and action space
#env = gym.make('Taxi-v2') # discrete state and action space

# Classic Control - Continuous State and Discrete Action Spaces
env = gym.make('MountainCar-v0') # needs Discretized or better
#env = gym.make('Acrobot-v1')      # needs Discretized, Tile Encoding or better
#env = gym.make('CartPole-v1')    # needs Deep Q Learning to do well?

# Classic Control - Continuous State and Action Spaces
#env = gym.make('Pendulum-v0') # continuous only
#env = gym.make('MountainCarContinuous-v0') # continuous only

# Box 2D
#env = gym.make('BipedalWalker-v2') # continuous only
#env = gym.make('CarRacing-v0')      # make the environment

# Atari
#env = gym.make('MsPacman-v0')

# Initialize the simulation
env.seed(505);
env.reset()
state = env.reset()

# Examine the environment
from visuals import examine_environment, examine_environment_MountainCar_discretized, examine_environment_Acrobat_tiled
#examine_environment(env)

state_size = env.observation_space.shape[0]
action_size = env.action_space.n
print("env.observation_space.shape[0]", state_size)
print("env.action_space", action_size)

"""
# Create Agent
"""

"""
# create the agent discretized state space Q Learning
"""
from agents import discretize as dis
from agents import QLearningAgentDiscretized as qlad
state_grid = dis.create_uniform_grid(env.observation_space.low, env.observation_space.high, bins=(20, 20))
agent = qlad.QLearningAgent(env, state_grid)

##examine_environment_MountainCar_discretized(env)

"""
# create the agent for tiled state space Q Learning
"""
#from agents import QLearningAgentDiscretizedTiles as qlat
#n_bins = 20
#bins = tuple([n_bins]*env.observation_space.shape[0])
#offset_pos = (env.observation_space.high - env.observation_space.low)/(3*n_bins)
#
#tiling_specs = [(bins, -offset_pos),
#                (bins, tuple([0.0]*env.observation_space.shape[0])),
#                (bins, offset_pos)]
#
#tq = qlat.TiledQTable(env.observation_space.low, 
#                 env.observation_space.high, 
#                 tiling_specs, 
#                 env.action_space.n)
#agent = qlat.QLearningAgentDisTiles(env, tq)


# Specialized enviroment examination
#examine_environment_Acrobat_tiled(env, n_bins)

"""
# Create Q network agent
"""



"""
# run the simulation
"""
import run as sim
num_episodes=25000
score = 0
scores = sim.run(agent, env, num_episodes, mode='train')

"""
# Plot scores obtained per episode
"""
from visuals import plot_scores, plot_q_table
rolling_mean = plot_scores(scores)

"""
# Run in test mode and analyze scores obtained
"""
test_scores = sim.run(agent, env, num_episodes=100, mode='test')
print("[TEST] Completed {} episodes with avg. score = {}".format(len(test_scores), np.mean(test_scores)))
_ = plot_scores(test_scores)

#plot_q_table(agent.q_table)

"""
# Watch Agent
"""
state = env.reset()
score = 0
for t in range(5000):
    # get action from agent
    action = agent.act(state)
       
    # show environment and step it forward
    env.render()
    state, reward, done, _ = env.step(action)
    score += reward
    if done:
        break 
print('Final score:', score)

"""
# Exit Environment
"""
#env.close()