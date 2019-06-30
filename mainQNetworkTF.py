# -*- coding: utf-8 -*-
"""
Created on Sun May 26 14:34:34 2019

@author: shane
"""


import sys
import gym
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

# Classic Control - Continuous State and Discrete Action Spaces
env = gym.make('MountainCar-v0') # needs Discretized or better
#env = gym.make('Acrobot-v1')      # needs Discretized, Tile Encoding or better
#env = gym.make('CartPole-v1')    # needs Deep Q Learning to do well?

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
Q Network Simulation
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


agent = qnet.QNetwork(name='main', state_size=state_size, action_size=action_size)

# Take one random step to get the pole and cart moving
state, reward, done, _ = env.step(env.action_space.sample())


# Now train with experiences
train_episodes = 2000          # max number of episodes to learn from
max_steps = 1000                # max steps in an episode
gamma = 0.99                   # future reward discount

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
#            env.render() 
            
            # Explore or Exploit
            # Exploration parameters
            explore_start = 1.0            # exploration probability at start
            explore_stop = 0.01            # minimum exploration probability 
            decay_rate = 0.0001            # exponential decay rate for exploration prob
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
                
                if ep > agent.pretrain_length:
                    print('Episode: {}'.format(ep),
                          'Total reward: {}'.format(total_reward),
                          'Training loss: {:.4f}'.format(loss),
                          'Explore P: {:.4f}'.format(explore_p))
                else:
                    print('Episode: {}'.format(ep),
                          'Total reward: {}'.format(total_reward),
                          'Explore P: {:.4f}'.format(explore_p))

                rewards_list.append((ep, total_reward))
                
                # Add experience to memory
                agent.memory.add((state, action, reward, next_state))
                
                # Start new episode
                env.reset()
                # Take one random step to get the pole and cart moving
                state, reward, done, _ = env.step(env.action_space.sample())

            else:
                # Add experience to memory
                agent.memory.add((state, action, reward, next_state))
                state = next_state
                t += 1
            

            # TODO check min memory size here? If big enough, train
            if ep > agent.pretrain_length:
                # Sample mini-batch from memory
                batch = agent.memory.sample(agent.batch_size)
                states = np.array([each[0] for each in batch])
                actions = np.array([each[1] for each in batch])
                rewards = np.array([each[2] for each in batch])
                next_states = np.array([each[3] for each in batch])
                
                # Train network
                target_Qs = sess.run(agent.output, feed_dict={agent.inputs_: next_states})
                
                # Set target_Qs to 0 for states where episode ends
                episode_ends = (next_states == np.zeros(states[0].shape)).all(axis=1)
                target_Qs[episode_ends] = 0#(0, 0) # CartPole-v1
#                target_Qs[episode_ends] = (0, 0)  # MountainCar-v0
                
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
    
    """
    # Exit Environment
    """
    env.close()  


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