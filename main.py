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

1. try the idea of sleeping the network with a different memory buffer size
2. try transfer learning between environments (batch norm to help?)

"""

# Create an environment and set random seed
#env = gym.make('MountainCar-v0') # needs Discretized or better
#env = gym.make('Acrobot-v1')      # needs Discretized, Tile Encoding or better
env = gym.make('CartPole-v1')    # needs Deep Q Learning to do well?
#env = gym.make('Taxi-v2') # discrete state and action space

#env = gym.make('Pendulum-v0') # continuous only
#env = gym.make('MountainCarContinuous-v0') # continuous only

env.seed(505);

state = env.reset()

# Examine the environment
from visuals import examine_environment, examine_environment_MountainCar_discretized, examine_environment_Acrobat_tiled
examine_environment(env)

"""
# create the agent discretized state space Q Learning
"""
#from agents import discretize as dis
#from agents import QLearningAgentDiscretized as qlad
#state_grid = dis.create_uniform_grid(env.observation_space.low, env.observation_space.high, bins=(20, 20))
#agent = qlad.QLearningAgent(env, state_grid)

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
#examine_environment_MountainCar_discretized(env)
#examine_environment_Acrobat_tiled(env, n_bins)

"""
# Create Q network agent
"""
from agents import QNetwork as qnet
import tensorflow as tf
if 'session' in locals() and session is not None:
    print('Close interactive session')
    session.close()
   
# Setup GPU TF stability
import tensorflow as tf
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
sess = tf.Session(config=tf.ConfigProto(
  allow_soft_placement=True, log_device_placement=True))
    
tf.reset_default_graph()
train_episodes = 1000          # max number of episodes to learn from
max_steps = 200                # max steps in an episode
gamma = 0.99                   # future reward discount

# Exploration parameters
explore_start = 1.0            # exploration probability at start
explore_stop = 0.01            # minimum exploration probability 
decay_rate = 0.0001            # exponential decay rate for exploration prob

# Network parameters
hidden_size = 64               # number of units in each Q-network hidden layer
learning_rate = 0.0001         # Q-network learning rate

# Memory parameters
memory_size = 10000            # memory capacity
batch_size = 20                # experience mini-batch size
pretrain_length = batch_size   # number experiences to pretrain the memory

agent = qnet.QNetwork(name='main', hidden_size=hidden_size, learning_rate=learning_rate)

# Initialize the simulation
env.reset()
# Take one random step to get the pole and cart moving
state, reward, done, _ = env.step(env.action_space.sample())

memory = qnet.Memory(max_size=memory_size)

# Make a bunch of random actions and store the experiences
for ii in range(pretrain_length):

    # Make a random action
    action = env.action_space.sample()
    next_state, reward, done, _ = env.step(action)

    if done:
        # The simulation fails so no next state
        next_state = np.zeros(state.shape)
        # Add experience to memory
        memory.add((state, action, reward, next_state))
        
        # Start new episode
        env.reset()
        # Take one random step to get the pole and cart moving
        state, reward, done, _ = env.step(env.action_space.sample())
    else:
        # Add experience to memory
        memory.add((state, action, reward, next_state))
        state = next_state


# Now train with experiences
saver = tf.train.Saver()
rewards_list = []
with tf.Session() as sess:
    # Initialize variables
    sess.run(tf.global_variables_initializer())
    
    step = 0
    for ep in range(1, train_episodes):
        total_reward = 0
        t = 0
        while t < max_steps:
            step += 1
            # Uncomment this next line to watch the training
            env.render() 
            
            # Explore or Exploit
            explore_p = explore_stop + (explore_start - explore_stop)*np.exp(-decay_rate*step) 
            if explore_p > np.random.rand():
                # Make a random action
                action = env.action_space.sample()
            else:
                # Get action from Q-network
                feed = {agent.inputs_: state.reshape((1, *state.shape))}
                Qs = sess.run(agent.output, feed_dict=feed)
                action = np.argmax(Qs)
            
            # Take action, get new state and reward
            next_state, reward, done, _ = env.step(action)
    
            total_reward += reward
            
            if done:
                # the episode ends so no next state
                next_state = np.zeros(state.shape)
                t = max_steps
                
                print('Episode: {}'.format(ep),
                      'Total reward: {}'.format(total_reward),
                      'Training loss: {:.4f}'.format(loss),
                      'Explore P: {:.4f}'.format(explore_p))
                rewards_list.append((ep, total_reward))
                
                # Add experience to memory
                memory.add((state, action, reward, next_state))
                
                # Start new episode
                env.reset()
                # Take one random step to get the pole and cart moving
                state, reward, done, _ = env.step(env.action_space.sample())

            else:
                # Add experience to memory
                memory.add((state, action, reward, next_state))
                state = next_state
                t += 1
            
            # Sample mini-batch from memory
            batch = memory.sample(batch_size)
            states = np.array([each[0] for each in batch])
            actions = np.array([each[1] for each in batch])
            rewards = np.array([each[2] for each in batch])
            next_states = np.array([each[3] for each in batch])
            
            # Train network
            target_Qs = sess.run(agent.output, feed_dict={agent.inputs_: next_states})
            
            # Set target_Qs to 0 for states where episode ends
            episode_ends = (next_states == np.zeros(states[0].shape)).all(axis=1)
            target_Qs[episode_ends] = (0, 0)
            
            targets = rewards + gamma * np.max(target_Qs, axis=1)

            loss, _ = sess.run([agent.loss, agent.opt],
                                feed_dict={agent.inputs_: states,
                                           agent.targetQs_: targets,
                                           agent.actions_: actions})
        
    saver.save(sess, "checkpoints/cartpole.ckpt")
    
    # Watch Agent
    state = env.reset()
    score = 0
    for t in range(3000):       
        # Get action from Q-network
        feed = {agent.inputs_: state.reshape((1, *state.shape))}
        Qs = sess.run(agent.output, feed_dict=feed)
        action = np.argmax(Qs)
        
        # show environment and step it forward
        env.render()
        state, reward, done, _ = env.step(action)
        score += reward
        if done:
            break 
    print('Final score:', score)


import matplotlib.pyplot as plt

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / N 

eps, rews = np.array(rewards_list).T
smoothed_rews = running_mean(rews, 10)
plt.plot(eps[-len(smoothed_rews):], smoothed_rews)
plt.plot(eps, rews, color='grey', alpha=0.3)
plt.xlabel('Episode')
plt.ylabel('Total Reward')

"""
# run the simulation
"""
#import run as sim
#num_episodes=25000
#score = 0
#scores = sim.run(agent, env, num_episodes)

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

plot_q_table(agent.q_table)

"""
# Watch Agent
"""
state = env.reset()
score = 0
for t in range(3000):
    # get action from agent
    action = agent.act(state, mode='test')
       
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
env.close()