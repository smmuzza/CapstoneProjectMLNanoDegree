# -*- coding: utf-8 -*-
"""
Created on Sat May 25 13:07:12 2019

@author: shane
"""

import gym


if __name__ == "__main__":
    env = gym.make("SpaceInvaders-v0")
    env.reset()
    for _ in range(1000):
        env.render()
        env.step(env.action_space.sample())