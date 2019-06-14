# -*- coding: utf-8 -*-
"""
Created on Sun May 19 12:55:20 2019

@author: shane
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from agents import discretize as dis

# Basic inspection of the environment
def examine_environment(env):

    # Run a random agent
    score = 0
    for t in range(200):
        action = env.action_space.sample()
        env.render()
        state, reward, done, _ = env.step(action)
        score += reward
        if done:
            break 
    print('Final score:', score)
    env.close()
    
    # Explore state (observation) space
    print("State space:", env.observation_space)
    print("- low:", env.observation_space.low)
    print("- high:", env.observation_space.high)
    
    # Generate some samples from the state space 
    print("State space samples:")
    print(np.array([env.observation_space.sample() for i in range(10)]))
    
    # Explore the action space
    print("Action space:", env.action_space)
    
    # Generate some samples from the action space
    print("Action space samples:")
    print(np.array([env.action_space.sample() for i in range(10)]))
    

# Initial Testing of environment and discretization
def examine_environment_MountainCar_discretized(env):

    # Run a random agent
    score = 0
    for t in range(200):
        action = env.action_space.sample()
        env.render()
        state, reward, done, _ = env.step(action)
        score += reward
        if done:
            break 
    print('Final score:', score)
    env.close()
    
    # Explore state (observation) space
    print("State space:", env.observation_space)
    print("- low:", env.observation_space.low)
    print("- high:", env.observation_space.high)
    
    # Generate some samples from the state space 
    print("State space samples:")
    print(np.array([env.observation_space.sample() for i in range(10)]))
    
    # Explore the action space
    print("Action space:", env.action_space)
    
    # Generate some samples from the action space
    print("Action space samples:")
    print(np.array([env.action_space.sample() for i in range(10)]))
    
    low = [-1.0, -5.0]
    high = [1.0, 5.0]
    dis.create_uniform_grid(low, high)  # [test]
    
    # Test with a simple grid and some samples
    grid = dis.create_uniform_grid([-1.0, -5.0], [1.0, 5.0])
    samples = np.array(
        [[-1.0 , -5.0],
         [-0.81, -4.1],
         [-0.8 , -4.0],
         [-0.5 ,  0.0],
         [ 0.2 , -1.9],
         [ 0.8 ,  4.0],
         [ 0.81,  4.1],
         [ 1.0 ,  5.0]])
    discretized_samples = np.array([dis.discretize(sample, grid) for sample in samples])
    print("\nSamples:", repr(samples), sep="\n")
    print("\nDiscretized samples:", repr(discretized_samples), sep="\n")
    
        
    dis.visualize_samples(samples, discretized_samples, grid, low, high)
    
    # Create a grid to discretize the state space
    state_grid = dis.create_uniform_grid(env.observation_space.low, env.observation_space.high, bins=(10, 10))
    state_grid
    
    # Obtain some samples from the space, discretize them, and then visualize them
    state_samples = np.array([env.observation_space.sample() for i in range(10)])
    discretized_state_samples = np.array([dis.discretize(sample, state_grid) for sample in state_samples])
    dis.visualize_samples(state_samples, discretized_state_samples, state_grid,
                      env.observation_space.low, env.observation_space.high)
    plt.xlabel('position'); plt.ylabel('velocity');  # axis labels for MountainCar-v0 state space

# Initial Testing of environment and discretization
from agents import tile
def examine_environment_Acrobat_tiled(env, n_bins):

    low = [-1.0, -5.0]
    high = [1.0, 5.0]
    tile.create_tiling_grid(low, high, bins=(n_bins, n_bins), offsets=(-0.1, 0.5))  # [test]
        
    # Tiling specs: [(<bins>, <offsets>), ...]
    tiling_specs = [((n_bins, n_bins), (-0.066, -0.33)),
                    ((n_bins, n_bins), (0.0, 0.0)),
                    ((n_bins, n_bins), (0.066, 0.33))]
    tilings = tile.create_tilings(low, high, tiling_specs)
        
    tile.visualize_tilings(tilings)
    
    
    # Test with some sample values
    samples = [(-1.2 , -5.1 ),
               (-0.75,  3.25),
               (-0.5 ,  0.0 ),
               ( 0.25, -1.9 ),
               ( 0.15, -1.75),
               ( 0.75,  2.5 ),
               ( 0.7 , -3.7 ),
               ( 1.0 ,  5.0 )]
    encoded_samples = [tile.tile_encode(sample, tilings) for sample in samples]
    print("\nSamples:", repr(samples), sep="\n")
    print("\nEncoded samples:", repr(encoded_samples), sep="\n")

    tile.visualize_encoded_samples(samples, encoded_samples, tilings);


def plot_scores(scores, rolling_window=50):
    """Plot scores and optional rolling mean using specified window."""
    plt.plot(scores); plt.title("Scores");
    rolling_mean = pd.Series(scores).rolling(rolling_window).mean()
    plt.plot(rolling_mean);
    return rolling_mean

def plot_q_table(q_table):
    """Visualize max Q-value for each state and corresponding action."""
    q_image = np.max(q_table, axis=2)       # max Q-value for each state
    q_actions = np.argmax(q_table, axis=2)  # best action for each state

    fig, ax = plt.subplots(figsize=(10, 10))
    cax = ax.imshow(q_image, cmap='jet');
    cbar = fig.colorbar(cax)
    for x in range(q_image.shape[0]):
        for y in range(q_image.shape[1]):
            ax.text(x, y, q_actions[x, y], color='white',
                    horizontalalignment='center', verticalalignment='center')
    ax.grid(False)
    ax.set_title("Q-table, size: {}".format(q_table.shape))
    ax.set_xlabel('position')
    ax.set_ylabel('velocity')