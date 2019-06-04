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
    best_reward = -np.inf                           # keep track of the best reward across episodes

    for i_episode in range(1, num_episodes+1):
        # Initialize episode
        state = env.reset()        # reset environment        
        agent.reset_episode(state) # reset agent      
        episode_steps = 0 # Reset for the new episode
        episode_total_reward = 0   # total rewards per episode
        done = False

        # Interact with the Environment in steps until done
        while not done:
            # 1. agent action given environment state
            #    assumes explore/exploit as part of agent design
            # 2. enviroment changes based on action
            # 3. (training mode) learn from environment feedback 
            #    (new state, reward, done) to agent
            # 4. step the agent (forward with the new state)
            action = agent.act(state)
            state, reward, done, info = env.step(action)
            
            if mode == 'train':
                agent.learn(action, reward, state, done)
            
            agent.step(state)

            episode_total_reward += reward
            episode_steps += 1

        # Save final score of the episode
        scores.append(episode_total_reward)
        
        # Print episode stats
        if mode == 'train':
#            if len(scores) > 100:
#                avg_score = np.mean(scores[-100:])
#                if avg_score > max_avg_score:
#                    max_avg_score = avg_score
#            if i_episode % 100 == 0:
#                print("\rEpisode {}/{} | Max Average Score: {}".format(i_episode, num_episodes, max_avg_score), end="")
#                sys.stdout.flush()
                
            if episode_total_reward > best_reward:
                best_reward = episode_total_reward 
                best_episode = i_episode
            print("\rEpisode = {:4d} (duration of {} steps); Reward = {:7.3f} (best reward = {:7.3f}, in episode {})   ".format(
                i_episode, episode_steps, episode_total_reward, best_reward, best_episode), end="")  # [debug]
            sys.stdout.flush()

    return scores