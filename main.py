# -*- coding: utf-8 -*-
"""
Created on Fri May 17 21:03:14 2019

@author: shane
"""

import gym
import numpy as np

np.set_printoptions(precision=3, linewidth=120)


"""
# Create an environment and set random seed
"""
selectedEnvironment = 6
env = 0
envName = 0

# Toy Text - Discrete state and action space
if selectedEnvironment == 0:
    envName = 'Taxi-v2'

# Classic Control - Continuous State and Discrete Action Spaces
elif selectedEnvironment == 1: 
    envName = 'MountainCar-v0' # needs Discretized or better
elif selectedEnvironment == 2: 
    envName = 'Acrobot-v1'     # needs Discretized, Tile Encoding or better
elif selectedEnvironment == 3: 
    envName = 'CartPole-v1'    # needs Deep Q Learning to do well?

# Box 2D - Continuous State, Discrete Actions
elif selectedEnvironment == 4: 
    envName = 'LunarLander-v2' # discrete actions, continuous state

# Classic Control - Continuous State and Action Spaces
elif selectedEnvironment == 5: 
    envName = 'Pendulum-v0' # continuous only
elif selectedEnvironment == 6: 
    envName = 'MountainCarContinuous-v0' # continuous only

# Box 2D - Continuous State and Action Spaces
elif selectedEnvironment == 7:
    envName = 'LunarLanderContinuous-v2' # continuous only
elif selectedEnvironment == 8: 
    envName = 'BipedalWalker-v2'  # continuous only

# Box 2D - Image State and Continuous Action Spaces   
elif selectedEnvironment == 9: 
    envName = 'CarRacing-v0'      #  image input, actions [steer, gas, brake]

# Initialize the environment
env = gym.make(envName)
env.reset()

# Set output file paths based on environment
file_output_train = envName + '_train.txt'       # file name for saved results
file_output_test = envName + '_test.txt'       # file name for saved results

from visuals import examine_environment, examine_environment_MountainCar_discretized, examine_environment_Acrobat_tiled
#examine_environment(env)

from datetime import datetime
FORMAT = '%Y%m%d%H%M%S'
file_output_train = datetime.now().strftime(FORMAT) + file_output_train

"""
# Create Agent
"""
agent = 0
selectedAgent = 2
if selectedAgent == 0:
    # create the agent discretized state space Q Learning
    from agents import QLearningAgentDiscretized as qlad
    agent = qlad.QLearningAgent(env)
#    examine_environment_MountainCar_discretized(env)

if selectedAgent == 1:
    # create the agent for tiled state space Q Learning
    from agents import QLearningAgentDiscretizedTiles as qlat
    agent = qlat.QLearningAgentDisTiles(env)
#    examine_environment_Acrobat_tiled(env, n_bins)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    print("env.observation_space.shape[0]", state_size)
    print("env.action_space", action_size)

if selectedAgent == 2:
    # Create DDPG network agent
    from agents.DDPG import DDPG
    agent = DDPG(env, "continousStateAction")  # continousStateAction imageStateContinuousAction
    obsSpace = env.observation_space.shape
    print("task.observation_space: ", obsSpace)


"""
# run the simulation
"""
import interact as sim
num_episodes=200
sim.interact(agent, env, num_episodes, mode='train', file_output=file_output_train)

"""
# Plot training scores obtained per episode
"""
from visuals import plot_scores, plot_q_table, plot_score_from_file
plot_score_from_file(file_output_train)
#plot_q_table(agent.q_table)


"""
# Run in test mode and analyze scores obtained
"""
print("[TEST] Training Done, now running tests...")
test_scores = sim.interact(agent, env, num_episodes=1, mode='test', file_output=file_output_test)
plot_score_from_file(file_output_test)

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
if 0:
    env.close()
