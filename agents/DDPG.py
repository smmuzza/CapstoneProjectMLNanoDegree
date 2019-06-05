import numpy as np
import copy

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu, theta, sigma):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state
    
##OU Noise method for this task
#def OUNoise():
#    theta = 0.15
#    sigma = 0.2
#    state = 0
#    while True:
#        yield state
#        state += -theta*state+sigma*np.random.randn()


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
#from keras import layers, models, optimizers
#from keras import backend as K
#from keras import regularizers

class Actor:
    """
    Actor (Policy) Model, using Deep Deterministic Policy Gradients
    or DDPG. An actor-critic method, but with the key idea that the
    underlying policy function used is deterministic in nature, with
    some noise added in externally to produce the desired stochasticity
    in actions taken.
    Algorithm originally presented in this paper:
    Lillicrap, Timothy P., et al., 2015. Continuous Control with Deep
    Reinforcement Learning
    https://arxiv.org/pdf/1509.02971.pdf
    """

    def __init__(self, state_size, action_size, action_low, action_high):
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
        self.build_model()

    def build_model(self):
        """Build an actor (policy) network that maps states -> actions."""
        # Define input layer (states)
        states = keras.layers.Input(shape=(self.state_size,), name='states')

        print("self.state_size", self.state_size)

#        net = keras.layers.Conv2D(filters=16, kernel_size=8, padding='same', activation='relu', 
#                                input_shape=(32, 32, 3))(states)
#        net = keras.layers.MaxPooling2D(pool_size=2)(net)
#        net = keras.layers.Conv2D(filters=32, kernel_size=4, padding='same', activation='relu')(net)
#        net = keras.layers.MaxPooling2D(pool_size=2)(net)
#        net = keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu')(net)
#        net = keras.layers.MaxPooling2D(pool_size=2)(net)
#        net = keras.layers.Dropout(0.3)(net)
#        net = keras.layers.Flatten()(net)
#        net = keras.layers.Dense(units=300, activation='elu')(net)


        # Kernel initializer with fan-in mode and scale of 1.0
        kernel_initializer = keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)

        # Add hidden layers
        net = keras.layers.Dense(units=400, activation='elu', kernel_initializer=kernel_initializer)(states)
        net = keras.layers.Dense(units=300, activation='elu', kernel_initializer=kernel_initializer)(net)

        # Add final output layer with sigmoid activation
        raw_actions = keras.layers.Dense(units=self.action_size, activation='sigmoid', name='raw_actions', kernel_initializer=kernel_initializer)(net)

        # Note that the raw actions produced by the output layer are in a [0.0, 1.0] range
        # (using a sigmoid activation function). So, we add another layer that scales each
        # output to the desired range for each action dimension. This produces a deterministic
        # action for any given state vector.
        actions = keras.layers.Lambda(lambda x: (x * self.action_range) + self.action_low,
            name='actions')(raw_actions)

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
        optimizer = keras.optimizers.Adam(lr=0.0001)
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = keras.backend.function(
            inputs=[self.model.input, action_gradients, keras.backend.learning_phase()],
            outputs=[],
            updates=updates_op)


class Critic:
    """Critic (Value) Model, using Deep Deterministic Policy Gradients
    or DDPG. An actor-critic method, but with the key idea that the
    underlying policy function used is deterministic in nature, with
    some noise added in externally to produce the desired stochasticity
    in actions taken.
    Algorithm originally presented in this paper:
    Lillicrap, Timothy P., et al., 2015. Continuous Control with Deep
    Reinforcement Learning
    https://arxiv.org/pdf/1509.02971.pdf
    """

    def __init__(self, state_size, action_size):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size
        self.build_model()

    def build_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # Define input layers. The critic model needs to map (state, action) pairs to
        # their Q-values. This is reflected in the following input layers.
        states = keras.layers.Input(shape=(self.state_size,), name='states')
        actions = keras.layers.Input(shape=(self.action_size,), name='actions')


#        net_states = keras.layers.Conv2D(filters=16, kernel_size=2, padding='same', activation='relu', 
#                                input_shape=(32, 32, 3))(states)
#        net_states = keras.layers.MaxPooling2D(pool_size=2)(net_states)
#        net_states = keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu')(net_states)
#        net_states = keras.layers.MaxPooling2D(pool_size=2)(net_states)
#        net_states = keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu')(net_states)
#        net_states = keras.layers.MaxPooling2D(pool_size=2)(net_states)
#        net_states = keras.layers.Dropout(0.3)(net_states)
#        net_states = keras.layers.Flatten()(net_states)
#        net_states = keras.layers.Dense(units=300, activation='elu')(net_states)
#
#        net_actions = keras.layers.Dense(units=400, activation='elu')(actions)

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

        # Add final output layer to produce action values (Q values). The final output
        # of this model is the Q-value for any given (state, action) pair.
        Q_values = keras.layers.Dense(units=1, activation=None, name='q_values', kernel_initializer=kernel_initializer)(net)

        # Create Keras model
        self.model = keras.models.Model(inputs=[states, actions], outputs=Q_values)

        # Define optimizer and compile model for training with built-in loss function
        # Use learning rate of 0.001
        optimizer = keras.optimizers.Adam(lr=0.001)
        self.model.compile(optimizer=optimizer, loss='mse')

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
        

        
class DDPG():
    """
    Reinforcement Learning agent that learns by using DDPG, or Deep 
    Deterministic Policy Gradients. An actor-critic method, but with 
    the key idea that the underlying policy function used is deterministic 
    in nature, with some noise added in externally to produce the desired 
    stochasticity in actions taken.
    Algorithm originally presented in this paper:
    Lillicrap, Timothy P., et al., 2015. Continuous Control with Deep 
    Reinforcement Learning
    https://arxiv.org/pdf/1509.02971.pdf
    
    Code in this class, as well as from the Actor, Critic, and 
    ReplayBuffer classes in model_ddpg_agent_mountain_car_continuous.py was 
    adopted from sample code that introduced DDPG in the Reinforcement Learning 
    lesson in Udacity's Machine Learning Engineer nanodegree.
    
    Certain modifications to the Udacity approach, such as using an 
    initial exploration policy to warm up (3 times longer than typical) a larger memory buffer 
    (batch size of 256 instead of 64) was inspired by another DDPG solution 
    to OpenAI Gym's 'MountainCarContinuous-v0' environment. This 
    implementation can be viewed at: 
    
    https://github.com/lirnli/OpenAI-gym-solutions/blob/master/Continuous_Deep_Deterministic_Policy_Gradient_Net/DDPG%20Class%20ver2.ipynb
    
    Note that we will need two copies of each model - one local and one target. 
    This is an extension of the "Fixed Q Targets" technique from Deep Q-Learning, 
    and is used to decouple the parameters being updated from the ones that are 
    producing target values.
    """
    
    def __init__(self, task):
        self.task = task

        # For OpenAI Gym envs, the following attributes need 
        # to be calculated differently from from a standard 
        # Quadcopter task.
        self.action_size = task.action_space.shape[0]
        self.action_low = task.action_space.low[0]
        self.action_high = task.action_space.high[0]
              
        # If task is OpenAi Gym 'MountainCarContinuous-v0' environment
        # Adjust state size to take advantage of action_repeat parameter.
        # Must do this here when running the 'MountainCarContinuous-v0' environment.
        self.action_repeat = 1
        self.state_size = task.observation_space.shape[0] * self.action_repeat

        print("task.observation_space: ", task.observation_space)

        # Actor (Policy) Model
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high)

        # Critic (Value) Model
        self.critic_local = Critic(self.state_size, self.action_size)
        self.critic_target = Critic(self.state_size, self.action_size)

        # Initialize target model parameters with local model parameters
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())

        # Replay memory
        self.buffer_size = 10000
        self.batch_size = 256
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

        # Algorithm parameters
        self.gamma = 0.99  # discount factor
        self.tau = 0.001   # for soft update of target parameters
        
        # Exploration Policy
                # Noise process (should be in units of RPMs for each rotor)
        self.exploration_mu = 0       # original 0
        self.exploration_theta = 0.15 # original 0.15
        self.exploration_sigma = 0.2  # oringal 0.2
        self.noise = OUNoise(self.action_size, self.exploration_mu, self.exploration_theta, self.exploration_sigma)
        
        self.i_episode = 0
        self.max_explore_eps = 100 # duration of exploration phase using OU noise
        self.stepCount = 0
        self.explore_p = 1.0

    def reset_episode(self, state):
#        state = self.task.reset()
        
        # Since the task is OpenAi Gym 'MountainCarContinuous-v0' environment, 
        # we must expand the state returned from the gym environment according to 
        # our chosen action_repeat parameter value.
        state = np.concatenate([state] * self.action_repeat) 
        
        self.last_state = state
        
        # increase episode counter
        self.i_episode += 1
        
        print("resetting episode... next explore_p: ", self.explore_p)
        
        return state

    def step(self, next_state):
        
        # Ensure that size of next_state as returned from the 
        # 'MountainCarContinuous-v0' environment is increased in 
        # size according to the action_repeat parameter's value.
        next_state = np.concatenate([next_state] * self.action_repeat) 

        # increase step count
        self.stepCount += 1

        # Roll over last state and action
        self.last_state = next_state
                
    def act(self, state):       
                 
        # Ensure that size of next_state as returned from the 
        # 'MountainCarContinuous-v0' environment is increased in 
        # size according to the action_repeat parameter's value.
        state = np.concatenate([state] * self.action_repeat) 
       
        # Exploration parameters
        explore_start = 1.0            # exploration probability at start
        explore_stop = 0.01            # minimum exploration probability 
        decay_rate = 0.00001            # exponential decay rate for exploration prob

            # exploration policy
            # TDOO put inside agent.act
#            if i_episode < max_explore_eps:
#                p = i_episode/max_explore_eps
#                nextNoise = next(noise)
#                action = action*p + (1-p)*nextNoise # Only a fraction of the action's value gets perturbed

        # Explore or Exploit
        self.explore_p = explore_stop + (explore_start - explore_stop)*np.exp(-decay_rate*self.stepCount) 
        if self.explore_p > np.random.rand():
            # Make a random action
            action = self.task.action_space.sample()
            
            # use a fraction of the explore p to randomly sample
            # this approach uses the network as momentum instead of 
            # simply using completely random actions    
        #                action = explore_p * env.action_space.sample() + (1 - explore_p) * agent.act(state)
        else:
            """Returns action(s) for given state(s) as per current policy."""
            state = np.reshape(state, [-1, self.state_size])
            action = self.actor_local.model.predict(state)[0]
        
        return action

    def learn(self, action, reward, next_state, done):
        
        # Ensure that size of next_state as returned from the 
        # 'MountainCarContinuous-v0' environment is increased in 
        # size according to the action_repeat parameter's value.
        next_state = np.concatenate([next_state] * self.action_repeat) 
        
        # Save experience / reward
        self.memory.add(self.last_state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size*3:    # Warm up period is 3 times longer than typical
            experiences = self.memory.sample(self.batch_size)
#            self.learn(experiences)
        
            """Update policy and value parameters using given batch of experience tuples."""
            # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
            states = np.vstack([e.state for e in experiences if e is not None])
            actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
            rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
            dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
            next_states = np.vstack([e.next_state for e in experiences if e is not None])
    
            # Get predicted next-state actions and Q values from target models
            actions_next = self.actor_target.model.predict_on_batch(next_states)
            Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])
    
            # Compute Q targets for current states and train critic model (local)
            Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
            self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)
    
            # Train actor model (local)
            action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
            self.actor_local.train_fn([states, action_gradients, 1])  # custom training function
    
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
