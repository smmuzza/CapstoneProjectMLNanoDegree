# Capstone Project - DDPG for OpenAI Gym

Completion Date: August 2019

Reinforcement learning project using OpenAI gym with a DDPG agent and others.

The following packages are required.
* Keras (https://keras.io/)
* OpenAI Gym (http://gym.openai.com)

$ pip install keras
$ git clone https://github.com/openai/gym (recommended to use this repo instead of apt-get)

In OpenAI Gym, the following Environments are supported. 
* 'MountainCar-v0' # needs Discretized agent
* 'Acrobot-v1'     # needs Discretized agent
* 'CartPole-v1'    # needs Discretized agent
* 'Pendulum-v0'    # continuous only
* 'MountainCarContinuous-v0' # continuous only
* 'LunarLanderContinuous-v2' # continuous only
* 'BipedalWalker-v2'  # continuous only
* 'CarRacing-v0'      #  image input, actions [steer, gas, brake]

All of these enviroments and 4 agent types (discretized A/B, Q Network, DDPG), can be chosen by editing options in main.py and running this script. The main script will select agent and environment, training the agent, test the age, and write training results and network weights to file based on time and date. 


