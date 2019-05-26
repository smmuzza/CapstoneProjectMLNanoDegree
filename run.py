# -*- coding: utf-8 -*-
"""
Created on Sun May 19 12:30:32 2019

@author: shane
"""
import sys
import numpy as np

def run(agent, env, num_episodes=20000, mode='train'):
    """Run agent in given reinforcement learning environment and return scores."""
    scores = []
    max_avg_score = -np.inf
    for i_episode in range(1, num_episodes+1):
        # Initialize episode
        state = env.reset() # reset environment        
        agent.reset_episode(state) # reset agent
        total_reward = 0
        done = False

        # Interact with the Environment in steps until done
        while not done:
            # 1. agent action given environment state
            # 2. enviroment changes based on action
            # 3. (training mode) learn from environment feedback 
            #    (new state, reward, done) to agent
            # 4. step the agent forward
            action = agent.act(state)
            state, reward, done, info = env.step(action)
            
            if mode == 'train':
                agent.learn(state, reward)
            
            agent.step(state)

            total_reward += reward

        # Save final score
        scores.append(total_reward)
        
        # Print episode stats
        if mode == 'train':
            if len(scores) > 100:
                avg_score = np.mean(scores[-100:])
                if avg_score > max_avg_score:
                    max_avg_score = avg_score
            if i_episode % 100 == 0:
                print("\rEpisode {}/{} | Max Average Score: {}".format(i_episode, num_episodes, max_avg_score), end="")
                sys.stdout.flush()

    return scores