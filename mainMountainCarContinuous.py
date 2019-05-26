# -*- coding: utf-8 -*-
"""
Created on Sat May 25 09:55:03 2019

@author: shane
"""
import gym
import numpy as np
from agents.DDPG import DDPG
import csv
import sys

#from gym import envs
#print(envs.registry.all())

#OU Noise method for this task
def OUNoise():
    theta = 0.15
    sigma = 0.2
    state = 0
    while True:
        yield state
        state += -theta*state+sigma*np.random.randn()

# Setup

# Toy Text - Discrete state and action space
#env = gym.make('Taxi-v2') # discrete state and action space

# Classic Control - Continuous State and Discrete Action Spaces
#env = gym.make('MountainCar-v0') # needs Discretized or better
#env = gym.make('Acrobot-v1')      # needs Discretized, Tile Encoding or better
#env = gym.make('CartPole-v1')    # needs Deep Q Learning to do well?

# Classic Control - Continuous State and Action Spaces
env = gym.make('Pendulum-v0') # continuous only
#env = gym.make('MountainCarContinuous-v0') # continuous only

# Box 2D
#env = gym.make('BipedalWalker-v2') # continuous only
#env = gym.make('CarRacing-v0')      # make the environment

# Atari
#env = gym.make('MsPacman-v0')


# Initialize the simulation with a random seed
env.seed(505);
env.reset()
state = env.reset()

# Examine the environment
from visuals import examine_environment, examine_environment_MountainCar_discretized, examine_environment_Acrobat_tiled
examine_environment(env)

agent = DDPG(env)
action_repeat = 3                               # my DDPG implementation uses action_repeat
num_episodes = 200
rewards_list = []                               # store the total rewards earned for each episode
best_reward = -np.inf                           # keep track of the best reward across episodes
max_explore_eps = 100                           # duration of exploration phase using OU noise
episode_steps = 0
noise = OUNoise()

# In order to save the simulation's reward results to a CSV file.
file_output = 'ddpg_agent_mountain_car_continuous_data.txt'       # file name for saved results
labels = ['episode', 'timestep', 'reward']

# Run the simulation, and save the results.
with open(file_output, 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(labels)
    # Begin the simulation by starting a new episode
    state = agent.reset_episode() 
    # Run the simulation for each episode.
    for i_episode in range(1, num_episodes+1):
        total_reward = 0
        while True:
            action = agent.act(state)
            
#            env.render()

            # exploration policy
            if i_episode < max_explore_eps:
                p = i_episode/max_explore_eps
                action = action*p + (1-p)*next(noise) # Only a fraction of the action's value gets perturbed
            
            next_state, reward, done, _ = env.step(action)
            # Ensure that size of next_state as returned from the 
            # 'MountainCarContinuous-v0' environment is increased in 
            # size according to the action_repeat parameter's value.
            next_state = np.concatenate([next_state] * action_repeat) 
            total_reward += reward
            agent.step(action, reward, next_state, done)
            
            # Save agent's rewards earned during each timestep of each episode 
            # of the simulation to the CSV file.
            to_write = [i_episode] + [episode_steps] + [reward]
            writer.writerow(to_write)
            # Increase episode timestep count by one.
            episode_steps += 1
            
            if done:
                rewards_list.append((i_episode, total_reward))
                if total_reward > best_reward:
                    best_reward = total_reward 
                    best_episode = i_episode
                print("\rEpisode = {:4d} (duration of {} steps); Reward = {:7.3f} (best reward = {:7.3f}, in episode {})   ".format(
                    i_episode, episode_steps, total_reward, best_reward, best_episode), end="")  # [debug]
                sys.stdout.flush()
                state = agent.reset_episode() # start a new episode
                episode_steps = 0 # Reset for the new episode
                break
            else:
                state = next_state




"""
# Exit Environment
"""
env.close()

# Load simulation results from the .csv file
import pandas as pd
import matplotlib.pyplot as plt
# Load simulation results from the .csv file
results = pd.read_csv('ddpg_agent_mountain_car_continuous_data.txt')

# Plot the reward


# Total rewards for each episode
episode_rewards_mean = results.groupby(['episode'])[['reward']].mean()
episode_rewards_sum = results.groupby(['episode'])[['reward']].sum()

smoothed_mean = episode_rewards_mean.rolling(25).mean() 
smoothed_sum = episode_rewards_sum.rolling(25).mean() 

#print(episode_rewards)
plt.figure(1)
plt.plot(episode_rewards_mean, label='mean rewards')
plt.plot(smoothed_mean, label='running mean')
plt.legend()
axes = plt.gca()
axes.set_ylim([-10,10])
plt.show()  

# plot the sum rewards
plt.figure(2)
plt.plot(episode_rewards_sum, label='sum rewards')
plt.plot(smoothed_sum, label='running mean')
plt.legend()
axes = plt.gca()
axes.set_ylim([-100,100])
plt.show()  



