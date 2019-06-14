import numpy as np

import random
from collections import namedtuple, deque

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size: maximum size of buffer
            batch_size: size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, batch_size=64):
        """Randomly sample a batch of experiences from memory."""
        return random.sample(self.memory, k=self.batch_size)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    
import keras    
from keras import layers, models, optimizers
#from keras import backend as K
from keras import regularizers

class Actor:
    """
    Actor (Policy) Model for DDPG
    """

    def __init__(self, state_size, action_size, action_low, action_high, netArch, learning_rate):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            action_low (array): Min value of each action dimension
            action_high (array): Max value of each action dimension
        """
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low
        self.netArch = netArch # network architecture selection
        self.learning_rate = learning_rate
        self.build_model()

        print("*** init actor ***")
        print("self.action_range: ", self.action_range)

    def build_model(self):
        """Build an actor (policy) network that maps states -> actions."""
        states = 0
        actions = 0

        def scale_output(x):
            temp = (x * np.array(self.action_range)) + np.array(self.action_low)
            return temp          
        
        
        if self.netArch =="Lillicrap":
            # Define input layer (states)
            states = keras.layers.Input(shape=(self.state_size,), name='states')
    
            # Kernel initializer with fan-in mode and scale of 1.0
            kernel_initializer = keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)
    
            # Add hidden layers
            net = keras.layers.Dense(units=400, activation='elu', kernel_initializer=kernel_initializer)(states)
            net = keras.layers.Dense(units=300, activation='elu', kernel_initializer=kernel_initializer)(net)
    
            # Add final output layer with sigmoid activation
            raw_actions = keras.layers.Dense(units=self.action_size, activation='sigmoid', name='raw_actions', kernel_initializer=kernel_initializer)(net)
        
        elif self.netArch == "QuadCopter":
            # Define input layer (states)
            states = keras.layers.Input(shape=(self.state_size,), name='states')
   
            net = layers.Dense(units=64, activation='relu', \
                   kernel_regularizer=regularizers.l2(0.001))(states)
            net = layers.Dropout(0.1)(net)

            net = layers.Dense(units=128, activation='relu', \
                   kernel_regularizer=regularizers.l2(0.001))(net)
            net = layers.Dropout(0.1)(net)

            net = layers.Dense(units=128, activation='relu', \
                   kernel_regularizer=regularizers.l2(0.001))(net)
            net = layers.Dropout(0.1)(net)

            net = layers.Dense(units=64, activation='relu', \
                   kernel_regularizer=regularizers.l2(0.001))(net)
            net = layers.Dropout(0.1)(net)

            # Add final output layer with sigmoid activation
            raw_actions = keras.layers.Dense(units=self.action_size, activation='sigmoid', name='raw_actions')(net)

        elif self.netArch == "QuadCopterBig":
            
            bigUp = 2
            
            # Define input layer (states)
            states = keras.layers.Input(shape=(self.state_size,), name='states')
   
            net = layers.Dense(units=64 * bigUp, activation='relu', \
                   kernel_regularizer=regularizers.l2(0.001))(states)
            net = layers.Dropout(0.1)(net)

            net = layers.Dense(units=128 * bigUp, activation='relu', \
                   kernel_regularizer=regularizers.l2(0.001))(net)
            net = layers.Dropout(0.1)(net)

            net = layers.Dense(units=128 * bigUp, activation='relu', \
                   kernel_regularizer=regularizers.l2(0.001))(net)
            net = layers.Dropout(0.1)(net)

            net = layers.Dense(units=64 * bigUp, activation='relu', \
                   kernel_regularizer=regularizers.l2(0.001))(net)
            net = layers.Dropout(0.1)(net)

            # Add final output layer with sigmoid activation
            raw_actions = keras.layers.Dense(units=self.action_size, activation='sigmoid', name='raw_actions')(net)

        elif self.netArch == "QuadCopterBatchNorm":
            # This network seems to produce more intelligable actions with less episodes, 
            # but is also significatly faster than without batch normalization
            # doesn't seem to be strong evidence that it trains faster, even with a 4x 
            # learning rate
            
            # Define input layer (states)
            states = keras.layers.Input(shape=(self.state_size,), name='states')
   
            net = layers.Dense(units=64, use_bias=False, activation=None, \
                   kernel_regularizer=regularizers.l2(0.001))(states)
            net = layers.BatchNormalization()(net) # (SMM) seems to help smooth results
            net = layers.Activation("relu")(net)
            net = layers.Dropout(0.1)(net)

            net = layers.Dense(units=128, use_bias=False, activation=None, \
                   kernel_regularizer=regularizers.l2(0.001))(net)
            net = layers.BatchNormalization()(net) # (SMM) seems to help smooth results
            net = layers.Activation("relu")(net)
            net = layers.Dropout(0.1)(net)

            net = layers.Dense(units=128, use_bias=False, activation=None, \
                   kernel_regularizer=regularizers.l2(0.001))(net)
            net = layers.BatchNormalization()(net) # (SMM) seems to help smooth results
            net = layers.Activation("relu")(net)
            net = layers.Dropout(0.1)(net)

            net = layers.Dense(units=64, use_bias=False, activation=None, \
                   kernel_regularizer=regularizers.l2(0.001))(net)
            net = layers.BatchNormalization()(net) # (SMM) seems to help smooth results
            net = layers.Activation("relu")(net)
            net = layers.Dropout(0.1)(net)

            # Add final output layer with sigmoid activation
            raw_actions = keras.layers.Dense(units=self.action_size, activation='sigmoid', name='raw_actions')(net)
        
        elif self.netArch == "imageInputV1":     

            # for img state space
            states = keras.layers.Input(shape=(96, 96, 3), name='states')
            net = keras.layers.Conv2D(32, (8, 8), strides=[4, 4], padding='same', activation='relu')(states)
            net = keras.layers.MaxPooling2D(pool_size=2)(net)
            net = keras.layers.Dropout(0.2)(net)
            net = keras.layers.Conv2D(64, (4, 4), strides=[2, 2], padding='same', activation='relu')(net)
            net = keras.layers.MaxPooling2D(pool_size=2)(net)
            net = keras.layers.Dropout(0.2)(net)
            net = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(net)
            net = keras.layers.MaxPooling2D(pool_size=2)(net)
            net = keras.layers.Dropout(0.2)(net)
            net = keras.layers.Flatten()(net)
            net = keras.layers.Dense(units=256, activation='relu')(net)
            net = keras.layers.Dropout(0.2)(net)
            net = keras.layers.Dense(units=256, activation='relu')(net)
            net = keras.layers.Dropout(0.2)(net)
            raw_actions = keras.layers.Dense(units=self.action_size, activation='sigmoid', name='raw_actions')(net)


        # Note that the raw actions produced by the output layer are in a [0.0, 1.0] range
        # (using a sigmoid activation function). So, we add another layer that scales each
        # output to the desired range for each action dimension. This produces a deterministic
        # action for any given state vector.
        actions = keras.layers.Lambda(lambda x: (x * self.action_range) + self.action_low, name='actions')(raw_actions)
#            actions = keras.layers.Lambda(lambda x: x, name='actions')(raw_actions)
#            actions = keras.layers.Lambda(scale_output, name='actions')(raw_actions)
#            actions = keras.layers.Lambda(scale_putsParallel, scale_putsParallel_shape, name='actions')(raw_actions)
           # SUSPECT THAT LAMBDA IS NOT SCALING PROPERLY FOR MULTI-D ACTIONS WITH DIFFERENT RANGES

        # Create Keras model
        self.model = keras.models.Model(inputs=states, outputs=actions)

        # Define loss function using action value (Q value) gradients
        # These gradients will need to be computed using the critic model, and
        # fed in while training. This is why they are specified as part of the
        # "inputs" used in the training function.
        action_gradients = keras.layers.Input(shape=(self.action_size,))
        loss = keras.backend.mean(-action_gradients * actions)

        # Define optimizer and training function
        # Use learning rate of 0.0001
        optimizer = keras.optimizers.Adam(lr=self.learning_rate) # amsgrad=True, 0.00025 ATARI paper learning rate
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = keras.backend.function(
            inputs=[self.model.input, action_gradients, keras.backend.learning_phase()],
            outputs=[],
            updates=updates_op)       

        print("\n\n*** ACTOR MODEL SUMMARY ***\n\n")
        self.model.summary()



class Critic:
    """
    Critic (Value) Model for DDPG
    """

    def __init__(self, state_size, action_size, netArch, learning_rate):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size
        self.netArch = netArch # network architecture selection
        self.learning_rate = learning_rate
        self.build_model()

    def build_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        states = 0
        actions = 0
        Q_values = 0
        
        if self.netArch =="Lillicrap":
            # Define input layers. The critic model needs to map (state, action) pairs to
            # their Q-values. This is reflected in the following input layers.
            states = keras.layers.Input(shape=(self.state_size,), name='states')
            actions = keras.layers.Input(shape=(self.action_size,), name='actions')
    
            # Kernel initializer with fan-in mode and scale of 1.0
            kernel_initializer = keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)
    
            # Add hidden layer(s) for state pathway
            net_states = keras.layers.Dense(units=400, activation='elu', kernel_initializer=kernel_initializer)(states)
    
            # Add hidden layer(s) for action pathway
            net_actions = keras.layers.Dense(units=400, activation='elu', kernel_initializer=kernel_initializer)(actions)
    
            # Combine state and action pathways. The two layers can first be processed via separate
            # "pathways" (mini sub-networks), but eventually need to be combined.
            net = keras.layers.Add()([net_states, net_actions])
    
            # Add more layers to the combined network if needed
            net = keras.layers.Dense(units=300, activation='elu', kernel_initializer=kernel_initializer)(net)
        
        elif self.netArch == "QuadCopter":
            
            # Define input layers. The critic model needs to map (state, action) pairs to
            # their Q-values. This is reflected in the following input layers.
            states = keras.layers.Input(shape=(self.state_size,), name='states')
            actions = keras.layers.Input(shape=(self.action_size,), name='actions')
            
            net_states = layers.Dense(units=64, activation='relu')(states)
            net_states = layers.Dropout(0.1)(net_states)
            
            net_states = layers.Dense(units=128, activation='relu')(net_states)
            net_states = layers.Dropout(0.1)(net_states)
       
            # Add hidden layer(s) for action pathway
            net_actions = layers.Dense(units=64, activation='relu')(actions)
            net_actions = layers.Dropout(0.1)(net_actions)
            
            net_actions = layers.Dense(units=128, activation='relu')(net_actions)
            net_actions = layers.Dropout(0.1)(net_actions)

            # Combine state and action pathways
            net = layers.Add()([net_states, net_actions])
            net = layers.Dense(units=32, activation='relu')(net)
   
        elif self.netArch == "QuadCopterBig":

            # takes longer to get good results than the regular sizes copter network
            bigUp = 2 
            
            # Define input layers. The critic model needs to map (state, action) pairs to
            # their Q-values. This is reflected in the following input layers.
            states = keras.layers.Input(shape=(self.state_size,), name='states')
            actions = keras.layers.Input(shape=(self.action_size,), name='actions')
            
            net_states = layers.Dense(units=64 * bigUp, activation='relu')(states)
            net_states = layers.Dropout(0.1)(net_states)
            
            net_states = layers.Dense(units=128 * bigUp, activation='relu')(net_states)
            net_states = layers.Dropout(0.1)(net_states)
       
            # Add hidden layer(s) for action pathway
            net_actions = layers.Dense(units=64 * bigUp, activation='relu')(actions)
            net_actions = layers.Dropout(0.1)(net_actions)
            
            net_actions = layers.Dense(units=128 * bigUp, activation='relu')(net_actions)
            net_actions = layers.Dropout(0.1)(net_actions)

            # Combine state and action pathways
            net = layers.Add()([net_states, net_actions])
            net = layers.Dense(units=32 * bigUp, activation='relu')(net)
    
        elif self.netArch == "QuadCopterBatchNorm":
            # Define input layers. The critic model needs to map (state, action) pairs to
            # their Q-values. This is reflected in the following input layers.
            states = keras.layers.Input(shape=(self.state_size,), name='states')
            actions = keras.layers.Input(shape=(self.action_size,), name='actions')
            
            # Add hidden layer(s) for state pathway
            net_states = layers.Dense(units=64, use_bias=False, activation=None, \
                   kernel_regularizer=regularizers.l2(0.001))(states)
            net_states = layers.BatchNormalization()(net_states) 
            net_states = layers.Activation("relu")(net_states)
            net_states = layers.Dropout(0.1)(net_states)
            
            net_states = layers.Dense(units=128, use_bias=False, activation=None, \
                   kernel_regularizer=regularizers.l2(0.001))(net_states)
            net_states = layers.BatchNormalization()(net_states)
            net_states = layers.Activation("relu")(net_states)
            net_states = layers.Dropout(0.1)(net_states)
       
            # Add hidden layer(s) for action pathway
            net_actions = layers.Dense(units=64, use_bias=False, activation=None, \
                   kernel_regularizer=regularizers.l2(0.001))(actions)
            net_actions = layers.BatchNormalization()(net_actions)
            net_actions = layers.Activation("relu")(net_actions)
            net_actions = layers.Dropout(0.1)(net_actions)
           
            net_actions = layers.Dense(units=128, use_bias=False, activation=None, \
                   kernel_regularizer=regularizers.l2(0.001))(net_actions)
            net_actions = layers.BatchNormalization()(net_actions)
            net_actions = layers.Activation("relu")(net_actions)
            net_actions = layers.Dropout(0.1)(net_actions)

            # Combine state and action pathways
            net = layers.Add()([net_states, net_actions])
            net = layers.Dense(units=32, use_bias=False, activation=None, \
                   kernel_regularizer=regularizers.l2(0.001))(net)
            net = layers.BatchNormalization()(net) #(SMM) 
            net = layers.Activation("relu")(net)
            net = layers.Dropout(0.1)(net)
    
        elif self.netArch == "imageInputV1":     
            # for img state space
            actions = keras.layers.Input(shape=(self.action_size,), name='actions')
            # Add hidden layer(s) for action pathway
            net_actions = keras.layers.Dense(units=512, activation='relu')(actions)
    
            states = keras.layers.Input(shape=(96, 96, 3), name='states')
            net_states = keras.layers.Conv2D(32, (8, 8), strides=[4, 4], padding='same', activation='relu')(states)
            net_states = keras.layers.MaxPooling2D(pool_size=2)(states)
            net_states = keras.layers.Dropout(0.1)(net_states)
            net_states = keras.layers.Conv2D(64, (4, 4), strides=[2, 2], padding='same', activation='relu')(net_states)
            net_states = keras.layers.MaxPooling2D(pool_size=2)(net_states)
            net_states = keras.layers.Dropout(0.1)(net_states)
            net_states = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(net_states)
            net_states = keras.layers.MaxPooling2D(pool_size=2)(net_states)
            net_states = keras.layers.Dropout(0.1)(net_states)
            net_states = keras.layers.Flatten()(net_states)
            net_states = keras.layers.Dense(units=256, activation='relu')(net_states)
            net_states = keras.layers.Dropout(0.1)(net_states)
    
            net = keras.layers.Add()([net_states, net_actions])
    
            # Add more layers to the combined network if needed
            net = keras.layers.Dense(units=256, activation='relu')(net)
            net = keras.layers.Dropout(0.1)(net)
             

        # Add final output layer to produce action values (Q values). The final output
        # of this model is the Q-value for any given (state, action) pair.
        Q_values = keras.layers.Dense(units=1, activation=None, name='q_values')(net)

        # Create Keras model
        self.model = keras.models.Model(inputs=[states, actions], outputs=Q_values)
        # Define optimizer and compile model for training with built-in loss function
        # Use learning rate of 0.001
        optimizer = keras.optimizers.Adam(lr=self.learning_rate) # 0.00025 Atari paper learning rate
        self.model.compile(optimizer=optimizer, loss='mse')

        print("\n\n*** CRITIC MODEL SUMMARY ***\n\n")
        self.model.summary()

        # Compute action gradients (derivative of Q values w.r.t. to actions). We also need
        # to compute the gradient of every Q-value with respect to its corresponding action
        # vector. This is needed for training the actor model.
        # This step needs to be performed explicitly.
        action_gradients = keras.backend.gradients(Q_values, actions)

        # Finally, a separate function needs to be defined to provide access to these gradients.
        # Define an additional function to fetch action gradients (to be used by actor model).
        self.get_action_gradients = keras.backend.function(
            inputs=[*self.model.input, keras.backend.learning_phase()],
            outputs=action_gradients)
        

"""
# Sets all pixel values to be between (0,1)
# Parameters:
# - image: A grayscale (nxmx1) or RGB (nxmx3) array of floats
# Outputs:
# - image rescaled so all pixels are between 0 and 1
"""
def unit_image(image):
    image = np.array(image)
    return np.true_divide(image, 255.0)
   
"""    
# scales the output actions from the network
# this is important for multi dimensional actions with different ranges and low/hgh values    
"""
def scale_output(x, action_range, action_low):
    temp = (np.array(x) * np.array(action_range)) + np.array(action_low)
    return temp  
           
    
class DDPG():
    """
    Reinforcement Learning agent using Deep Deterministic Policy Gradients. 
    
    This is an actor-critic method, where the policy function used is deterministic 
    in nature, with some noise added in externally to produce the desired 
    stochasticity in actions taken.
    
    Original Paper:
    Lillicrap, Timothy P., et al., 2015. Continuous Control with Deep 
    Reinforcement Learning, https://arxiv.org/pdf/1509.02971.pdf
    
    Actor, Critic, and ReplayBuffer are based on sample code from the Quadcopter Project 
    in Udacity's Machine Learning Engineer nanodegree.
    
    Benchmark Implementation for OpenAI "MountainCarContinuous-v0" (DDPG): 
    https://github.com/lirnli/OpenAI-gym-solutions/blob/master/Continuous_Deep_Deterministic_Policy_Gradient_Net/DDPG%20Class%20ver2.ipynb
    
    Benchmark Implementation for OpenAI "Car Racing" (DDQN):
    https://github.com/AMD-RIPS/RL-2018/blob/master/documents/leaderboard/IPAM-AMD-Car_Racing.ipynb
    
    Note that we will need two copies of each model - one local and one target. 
    This is an extension of the "Fixed Q Targets" technique from Deep Q-Learning, 
    and is used to decouple the parameters being updated from the ones that are 
    producing target values.
    """
    
    def __init__(self, env, envType):
        self.env = env

        print("Initializing DDPG Agent")
        print("\tEnvironment: ", env)

        # For OpenAI Gym envs, the following attributes need 
        # to be calculated differently from from a standard 
        # Quadcopter env.
        self.action_size = env.action_space.shape[0]
        self.action_low = env.action_space.low       # can be an array depending on the environment
        self.action_high = env.action_space.high     # can be an array depending on the environment
        self.action_range = self.action_high - self.action_low  # can be an array depending on the environment

        print("env.action_space.shape", env.action_space.shape)
        print("env.action_space.low", env.action_space.low)
        print("env.action_space.high", env.action_space.high)
               
        # Set the typo of OpenAI Enviroment
        # Currently continuousStateAction, and imageStateContinuousAction are supported
        self.envType = envType

        # Action Repeat
        self.action_repeat = 1
        self.state_size = env.observation_space.shape[0] * self.action_repeat

        # select network based on enviromnet type
        self.learningRate = 0.0001
        network_arch = "QuadCopterBig" # QuadCopter, QuadCopterBig, QuadCopterBatchNorm, Lillicrap
        if envType == "imageStateContinuousAction":
            network_arch = "imageInputV1"

        print("DDPG network architecture chosen: ", network_arch)

        # Actor (Policy) Model
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high, network_arch, self.learningRate)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high, network_arch, self.learningRate)

        # Critic (Value) Model
        self.critic_local = Critic(self.state_size, self.action_size, network_arch, self.learningRate)
        self.critic_target = Critic(self.state_size, self.action_size, network_arch, self.learningRate)

        # Initialize target model parameters with local model parameters
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())

        """
        solve mt climber with copter network
            1. 75 episodes (unstable solution) with batch size/buffer 32/1024, gamma/tau .99/.01
               learning rate 0.0001, explore decay 0.00001, action repeat 1
            2. 140 episodes (unstable solution) with batch size/buffer 128/1024, gamma/tau .995/.005
               learning rate 0.0001, explore decay 0.00001, action repeat 1
            3. 90 episodes (stable solution) with batch size/buffer 256/10000, gamma/tau .995/.005
               learning rate 0.0001, explore decay 0.00001, action repeat 1
               
        xxx solve mt climber with copter BIG network
            1. 100 episodes (stable solution) with batch size/buffer 256/10000, gamma/tau .995/.005
               learning rate 0.0001, explore decay 0.00001, action repeat 1, batch norm

        cannot solve mt climber with copter batch norm network
            1. ~100 episodes (no solution) with batch size/buffer 256/10000, gamma/tau .995/.005
               learning rate 0.0001, explore decay 0.00001, action repeat 1, batch norm
        """

        # Replay memory
        # a large batch size seems to make the agent learn too much too fast without new data points
        # and it seems like this can cause the gradient decent to overshoot
        # in copter project also had better results with smaller batch sizes
        # probably the learning rates, gamma, and tau need to be adjusted down as the 
        # batch size goes up
        # so when in doubt, use a smaller batch size to see if learning happens, then 
        # increase to speed up the learning until it breaks, then tuning hyperparameters
        if self.envType == "continousStateAction":
            self.buffer_size = 10000 # most episodes are around 1000 steps in OpenAI for a complete run 
            self.batch_size = 256 
        elif self.envType == "imageStateContinuousAction":
            self.buffer_size = 10000 
            self.batch_size = 64
        else:    
            raise("\nDDPG:__init__: ERROR! unsupported env type!\n")            
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

        # Algorithm parameters
        self.gamma = 0.995 #0.99  # discount factor
        self.tau = 0.005 #0.01   # for soft update of target parameters
    
        # Exploration Policy (expodential decay based on lifetime steps)
        self.explore_start = 1.0            # exploration probability at start
        self.explore_stop = 0.001           # minimum exploration probability 
        self.decay_rate = 0.00001           # exponential decay rate for exploration prob
        self.exploreStep = 0
        self.explore_p = 1.0

        # step and episode counters   
        self.stepCount = 0
        self.i_episode = 0

        # Save the Q targets and action gradients for analysis and avoid memory stack allocations
        self.Q_targets = 0
        self.action_gradients = 0

    def reset_episode(self, state):
        
        # Since the env is OpenAi Gym 'MountainCarContinuous-v0' environment, 
        # we must expand the state returned from the gym environment according to 
        # our chosen action_repeat parameter value.
        state = np.concatenate([state] * self.action_repeat) 
        
        if self.envType == "imageStateContinuousAction":
            state = unit_image(state) # normalize to between 0-1
        
        self.last_state = state
        
        # increase episode counters
        self.i_episode += 1
        
        print("resetting episode... next explore_p: ", self.explore_p)
        
        return state

    def step(self, next_state):
        
        # Ensure that size of next_state as returned from the 
        # environment is increased in according to the action_repeat
        next_state = np.concatenate([next_state] * self.action_repeat) 
        
        if self.envType == "imageStateContinuousAction":
            next_state = unit_image(next_state)
        
        # increase step count
        self.stepCount += 1
        self.exploreStep += 1

        # Roll over last state and action
        self.last_state = next_state

                
    def act(self, state):       
                 
        # Ensure that size of next_state as returned from the 
        # environment is increased in according to the action_repeat
        state = np.concatenate([state] * self.action_repeat) 
        
        if self.envType == "imageStateContinuousAction":
            state = unit_image(state)
       
        # return a random action if memory is not filled (inital network weights)
        if len(self.memory) < self.batch_size:
            return self.env.action_space.sample()
        
        # Explore or Exploit
        # Use expodentially decaying noise, more consistant results across environments than OU noise
        self.explore_p = self.explore_stop + (self.explore_start - self.explore_stop)*np.exp(-self.decay_rate*self.exploreStep) 
        if self.explore_p > np.random.rand():
            # Make a random action
            action = self.env.action_space.sample()
            
#            if self.envType == "continousStateAction":
#                state = np.reshape(state, [-1, self.state_size])
#            elif self.envType == "imageStateContinuousAction":
#                state = np.expand_dims(state, axis=0)  # for img state space
#
#            # make an agent action proportional to explore_p
#            agentAction = self.actor_local.model.predict(state)[0]                 
#            randAction = self.env.action_space.sample()
#            action = agentAction * (1-self.explore_p) + randAction * self.explore_p            
            
        else:
            """Returns action(s) for given state(s) as per current policy."""
            if self.envType == "continousStateAction":
                state = np.reshape(state, [-1, self.state_size])
            elif self.envType == "imageStateContinuousAction":
                state = np.expand_dims(state, axis=0)  # for img state space

            action = self.actor_local.model.predict(state)[0]    
#            print("action before: ", action)
#            action = scale_output(action, self.action_range, self.action_low)
#            print("action after: ", action)           

            # Making the actions partially random seems to hurt the agent in finding the right actions
            # probably the correlation between the agent weights and state and actions gets
            # messed up and is not consistant
            # make an agent action proportional to 1 - explore_p
#            agentAction = self.actor_local.model.predict(state)[0]                 
#            randAction = self.env.action_space.sample()
#            action = agentAction * (1-self.explore_p) + randAction * self.explore_p            
#            print("\tagent steps cnt: ", self.stepCount, ", with action:", action)
        
        # cycle the exploration policy between explore and exploit
        # this helps the agent get unstuck follow bad actions over and over
#        if self.explore_p < self.explore_stop * 2:
#            self.exploreStep = 1000           
        
        return action

    def learn(self, action, reward, next_state, done):
        
        # Ensure that size of next_state as returned from the 
        # environment is increased in 
        # size according to the action_repeat parameter's value.
        next_state = np.concatenate([next_state] * self.action_repeat) 
        
        if self.envType == "imageStateContinuousAction":
            next_state = unit_image(next_state)
        
        # Save experience / reward
        self.memory.add(self.last_state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        # Check step count to avoid loading and unloading the GPU all the time
        if len(self.memory) > self.batch_size: # and self.stepCount % 1 == 0:
#            print("\t\tlearning on total training step count: ", self.stepCount)
            experiences = self.memory.sample(self.batch_size)
        
            """Update policy and value parameters using given batch of experience tuples."""
            # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
            states = np.vstack([e.state for e in experiences if e is not None])
            actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
            rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
            dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
            next_states = np.vstack([e.next_state for e in experiences if e is not None])
   
    	    # turn the states and next_states into numpy arrays 
            # this is important to properly stack image states, as vstack won't work properly on multiple dimensions	
            states = []
            for e in experiences:
                states.append(e.state)
    
            states = np.array(states)

            next_states = []
            for e in experiences:
                next_states.append(e.next_state)
    
            next_states = np.array(next_states)
    
            # Get predicted next-state actions and Q values from target models               
            actions_next = self.actor_target.model.predict_on_batch(next_states)
#            for act in actions_next:
#                act = scale_output(act, self.action_range, self.action_low)
            Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])
    
            # Compute Q targets for current states and train critic model (local)
            self.Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
            self.critic_local.model.train_on_batch(x=[states, actions], y=self.Q_targets)
    
            # Train actor model (local)
            self.action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
            self.actor_local.train_fn([states, self.action_gradients, 1])  # custom training function
    
            # Soft-update target models
            self.soft_update(self.critic_local.model, self.critic_target.model)
            self.soft_update(self.actor_local.model, self.actor_target.model)   

    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        assert len(local_weights) == len(target_weights), "Local and target model parameters must have the same size"

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)   
