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

    def __init__(self, state_size, action_size, action_low, action_high, netArch, learning_rate, dropout_rate):
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
        self.dropout_rate = dropout_rate
        self.build_model()

        print("*** init actor ***")
        print("self.action_range: ", self.action_range)

    def build_model(self):
        """Build an actor (policy) network that maps states -> actions."""
        states = 0
        actions = 0
        
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

        elif self.netArch =="UniformLayers":
            # Define input layer (states)
            states = keras.layers.Input(shape=(self.state_size,), name='states')

            layerSize = 300
            print("UniformLaters: layerSize", layerSize)
            # Define input layer (states)
#            kernel_initializer = keras.initializers.glorot_normal(seed=None)
       
            # Add hidden layers
            net = keras.layers.Dense(units=layerSize, activation='relu')(states)
            net = keras.layers.Dense(units=layerSize, activation='relu')(net)
            net = keras.layers.Dense(units=layerSize, activation='relu')(net)
            net = keras.layers.Dense(units=layerSize, activation='relu')(net)
            net = keras.layers.Dense(units=layerSize, activation='relu')(net)
            net = keras.layers.Dense(units=layerSize, activation='relu')(net)
    
            # Add final output layer with sigmoid activation
            raw_actions = keras.layers.Dense(units=self.action_size, activation='sigmoid', name='raw_actions')(net)
        
        elif self.netArch == "Hausknecht":

            # Define input layer (states)
            states = keras.layers.Input(shape=(self.state_size,), name='states')
      
            # Add hidden layers
#            net = keras.layers.Dense(units=32, activation='relu')(net)
#            net = keras.layers.Dense(units=64, activation='relu')(net)
            net = keras.layers.Dense(units=1024, activation='relu')(states)
            net = keras.layers.Dense(units=512, activation='relu')(net)
            net = keras.layers.Dense(units=256, activation='relu')(net)
            net = keras.layers.Dense(units=128, activation='relu')(net)
    
            # Add final output layer with sigmoid activation
            raw_actions = keras.layers.Dense(units=self.action_size, activation='sigmoid', name='raw_actions')(net)

        
        elif self.netArch == "QuadCopter":
            # Define input layer (states)
            states = keras.layers.Input(shape=(self.state_size,), name='states')
   
            net = layers.Dense(units=64, activation='relu', \
                   kernel_regularizer=regularizers.l2(0.001))(states)
            net = layers.Dropout(self.dropout_rate)(net)

            net = layers.Dense(units=128, activation='relu', \
                   kernel_regularizer=regularizers.l2(0.001))(net)
            net = layers.Dropout(self.dropout_rate)(net)

            net = layers.Dense(units=128, activation='relu', \
                   kernel_regularizer=regularizers.l2(0.001))(net)
            net = layers.Dropout(self.dropout_rate)(net)

            net = layers.Dense(units=64, activation='relu', \
                   kernel_regularizer=regularizers.l2(0.001))(net)
            net = layers.Dropout(self.dropout_rate)(net)

            # Add final output layer with sigmoid activation
            raw_actions = keras.layers.Dense(units=self.action_size, activation='sigmoid', name='raw_actions')(net)

        elif self.netArch == "QuadCopterBig":
            
            bigUp = 2 # option kernel_regularizer=regularizers.l2(0.01)
            
            # Define input layer (states)
            states = keras.layers.Input(shape=(self.state_size,), name='states')

#            noiseStd = 0.1
#            print("actor noiseStd: ", noiseStd)
#            net = keras.layers.GaussianNoise(noiseStd)(states)
   
            net = layers.Dense(units=64 * bigUp, activation='relu')(states)
            net = layers.Dropout(self.dropout_rate)(net)

            net = layers.Dense(units=128 * bigUp, activation='relu')(net)
            net = layers.Dropout(self.dropout_rate)(net)

            net = layers.Dense(units=128 * bigUp, activation='relu')(net)
            net = layers.Dropout(self.dropout_rate)(net)

            net = layers.Dense(units=64 * bigUp, activation='relu')(net)
            net = layers.Dropout(self.dropout_rate)(net)

            # Add final output layer with sigmoid activation
            raw_actions = keras.layers.Dense(units=self.action_size, activation='sigmoid', name='raw_actions')(net)

        elif self.netArch == "QuadCopterBigNoDropout":
            
            bigUp = 2            
            # Define input layer (states)
            states = keras.layers.Input(shape=(self.state_size,), name='states')   
            net = layers.Dense(units=64 * bigUp, activation='relu')(states)
            net = layers.Dense(units=128 * bigUp, activation='relu')(net)
            net = layers.Dense(units=128 * bigUp, activation='relu')(net)
            net = layers.Dense(units=64 * bigUp, activation='relu')(net)

            # Add final output layer with sigmoid activation
            raw_actions = keras.layers.Dense(units=self.action_size, activation='sigmoid', name='raw_actions')(net)
           
        elif self.netArch == "QuadCopterMax":
            
            bigUp = 3            
            # Define input layer (states)
            states = keras.layers.Input(shape=(self.state_size,), name='states')
   
            net = layers.Dense(units=64 * bigUp, activation='relu', \
                   kernel_regularizer=regularizers.l2(0.001))(states)
            net = layers.Dropout(self.dropout_rate)(net)

            net = layers.Dense(units=128 * bigUp, activation='relu', \
                   kernel_regularizer=regularizers.l2(0.001))(net)
            net = layers.Dropout(self.dropout_rate)(net)

            net = layers.Dense(units=128 * bigUp, activation='relu', \
                   kernel_regularizer=regularizers.l2(0.001))(net)
            net = layers.Dropout(self.dropout_rate)(net)

            net = layers.Dense(units=64 * bigUp, activation='relu', \
                   kernel_regularizer=regularizers.l2(0.001))(net)
            net = layers.Dropout(self.dropout_rate)(net)

            # Add final output layer with sigmoid activation
            raw_actions = keras.layers.Dense(units=self.action_size, activation='sigmoid', name='raw_actions')(net)

        elif self.netArch == "QuadCopterBigELU":
            
            bigUp = 2            
            # Define input layer (states)
            states = keras.layers.Input(shape=(self.state_size,), name='states')
            net = layers.Dense(units=64 * bigUp, activation='elu')(states)
            net = layers.Dense(units=128 * bigUp, activation='elu')(net)
            net = layers.Dense(units=128 * bigUp, activation='elu')(net)
            net = layers.Dense(units=64 * bigUp, activation='elu')(net)

            # Add final output layer with sigmoid activation
            raw_actions = keras.layers.Dense(units=self.action_size, activation='sigmoid', name='raw_actions')(net)
            
        elif self.netArch == "QuadCopterBatchNorm":
            # This network seems to produce more intelligable actions with less episodes, 
            # but is also significatly faster than without batch normalization
            # doesn't seem to be strong evidence that it trains faster, even with a 4x 
            # learning rate
            
            bigUp = 2            
            # Define input layer (states)
            states = keras.layers.Input(shape=(self.state_size,), name='states')
   
            net = layers.Dense(units=64 * bigUp, use_bias=False, activation=None)(states)
            net = layers.BatchNormalization()(net) # (SMM) seems to help smooth results
            net = layers.Activation("relu")(net)

            net = layers.Dense(units=128 * bigUp, use_bias=False, activation=None)(net)
            net = layers.BatchNormalization()(net) # (SMM) seems to help smooth results
            net = layers.Activation("relu")(net)

            net = layers.Dense(units=128 * bigUp, use_bias=False, activation=None)(net)
            net = layers.BatchNormalization()(net) # (SMM) seems to help smooth results
            net = layers.Activation("relu")(net)

            net = layers.Dense(units=64 * bigUp, use_bias=False, activation=None)(net)
            net = layers.BatchNormalization()(net) # (SMM) seems to help smooth results
            net = layers.Activation("relu")(net)

            # Add final output layer with sigmoid activation
            raw_actions = keras.layers.Dense(units=self.action_size, activation='sigmoid', name='raw_actions')(net)
            
        elif self.netArch == "QuadCopterBatchNormInput":
            # This network seems to produce more intelligable actions with less episodes, 
            # but is also significatly faster than without batch normalization
            # doesn't seem to be strong evidence that it trains faster, even with a 4x 
            # learning rate
            
            bigUp = 2            
            # Define input layer (states)
            states = keras.layers.Input(shape=(self.state_size,), name='states')

            net = layers.BatchNormalization()(states) # (SMM) seems to help smooth results
   
            net = layers.Dense(units=64 * bigUp, activation='relu')(net)
            net = layers.Dropout(self.dropout_rate)(net)

            net = layers.Dense(units=128 * bigUp, activation='relu')(net)
            net = layers.Dropout(self.dropout_rate)(net)

            net = layers.Dense(units=128 * bigUp, activation='relu')(net)
            net = layers.Dropout(self.dropout_rate)(net)

            net = layers.Dense(units=64 * bigUp, activation='relu')(net)
            net = layers.Dropout(self.dropout_rate)(net)

            # Add final output layer with sigmoid activation
            raw_actions = keras.layers.Dense(units=self.action_size, activation='sigmoid', name='raw_actions')(net)
                
        elif self.netArch == "imageInputV1":     

            # for img state space
            states = keras.layers.Input(shape=(96, 96, 3), name='states')
            net = keras.layers.Conv2D(32, (8, 8), strides=[4, 4], padding='same', activation='relu')(states)
            net = keras.layers.MaxPooling2D(pool_size=2)(net)
            net = keras.layers.Dropout(self.dropout_rate)(net)
            net = keras.layers.Conv2D(64, (4, 4), strides=[2, 2], padding='same', activation='relu')(net)
            net = keras.layers.MaxPooling2D(pool_size=2)(net)
            net = keras.layers.Dropout(self.dropout_rate)(net)
            net = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(net)
            net = keras.layers.MaxPooling2D(pool_size=2)(net)
            net = keras.layers.Dropout(self.dropout_rate)(net)
            net = keras.layers.Flatten()(net)
            net = keras.layers.Dense(units=512, activation='relu')(net)
            net = keras.layers.Dropout(self.dropout_rate)(net)
            net = keras.layers.Dense(units=256, activation='relu')(net)
            net = keras.layers.Dropout(self.dropout_rate)(net)
            raw_actions = keras.layers.Dense(units=self.action_size, activation='sigmoid', name='raw_actions')(net)


        # Note that the raw actions produced by the output layer are in a [0.0, 1.0] range
        # (using a sigmoid activation function). So, we add another layer that scales each
        # output to the desired range for each action dimension. This produces a deterministic
        # action for any given state vector.
#        actions = keras.layers.Lambda(lambda x: (x * self.action_range) + self.action_low, name='actions')(raw_actions)
#        actions = keras.layers.Lambda(lambda x: x, name='actions')(raw_actions)
        actions = raw_actions
        self.model = keras.models.Model(inputs=states, outputs=actions)
#            actions = keras.layers.Lambda(scale_output, name='actions')(raw_actions)
#            actions = keras.layers.Lambda(scale_putsParallel, scale_putsParallel_shape, name='actions')(raw_actions)
           # SUSPECT THAT LAMBDA IS NOT SCALING PROPERLY FOR MULTI-D ACTIONS WITH DIFFERENT RANGES

        # Create Keras model
#        self.model = keras.models.Model(inputs=states, outputs=actions)

        # Define loss function using action value (Q value) gradients
        # These gradients will need to be computed using the critic model, and
        # fed in while training. This is why they are specified as part of the
        # "inputs" used in the training function.
        action_gradients = keras.layers.Input(shape=(self.action_size,))
        loss = keras.backend.mean(-action_gradients * actions)

        # Define optimizer and training function
        optimizer = keras.optimizers.Adam(lr=self.learning_rate) # amsgrad=True
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = keras.backend.function(
            inputs=[self.model.input, action_gradients, keras.backend.learning_phase()],
            outputs=[],
            updates=updates_op)       


class Critic:
    """
    Critic (Value) Model for DDPG
    """

    def __init__(self, state_size, action_size, netArch, learning_rate, dropout_rate):
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
        self.dropout_rate = dropout_rate
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
        
        elif self.netArch =="UniformLayers":
            # Define input layers. The critic model needs to map (state, action) pairs to
            # their Q-values. This is reflected in the following input layers.
            states = keras.layers.Input(shape=(self.state_size,), name='states')
            actions = keras.layers.Input(shape=(self.action_size,), name='actions')
    
            # Kernel initializer Xavier
#            kernel_initializer = keras.initializers.glorot_normal(seed=None)
    
            layerSize = 300
            print("UniformLaters: layerSize", layerSize)
            
            # Add hidden layer(s) for state pathway
            net_states = keras.layers.Dense(units=layerSize, activation='relu')(states)
            net_states = keras.layers.Dense(units=layerSize, activation='relu')(net_states)
            net_states = keras.layers.Dense(units=layerSize, activation='relu')(net_states)
    
            # Add hidden layer(s) for action pathway
            net_actions = keras.layers.Dense(units=layerSize, activation='relu')(actions)
            net_actions = keras.layers.Dense(units=layerSize, activation='relu')(net_actions)
            net_actions = keras.layers.Dense(units=layerSize, activation='relu')(net_actions)
    
            # Combine state and action pathways. The two layers can first be processed via separate
            # "pathways" (mini sub-networks), but eventually need to be combined.
            net = keras.layers.Add()([net_states, net_actions])
    
            # Add more layers to the combined network if needed
            net = keras.layers.Dense(units=layerSize, activation='relu')(net)
        
        elif self.netArch == "Hausknecht":
            
            # Define input layers. The critic model needs to map (state, action) pairs to
            # their Q-values. This is reflected in the following input layers.
            states = keras.layers.Input(shape=(self.state_size,), name='states')
            actions = keras.layers.Input(shape=(self.action_size,), name='actions')
    
            # Combine state and action pathways. The two layers can first be processed via separate
            # "pathways" (mini sub-networks), but eventually need to be combined.
            net = keras.layers.Add()([states, actions])
     
            # Add hidden layer(s) for state pathway
            net = keras.layers.Dense(units=1024, activation='relu')(net)
            net = keras.layers.Dense(units=512, activation='relu')(net)
            net = keras.layers.Dense(units=256, activation='relu')(net)
            net = keras.layers.Dense(units=128, activation='relu')(net)   
                 
        elif self.netArch == "QuadCopter":
            
            # Define input layers. The critic model needs to map (state, action) pairs to
            # their Q-values. This is reflected in the following input layers.
            states = keras.layers.Input(shape=(self.state_size,), name='states')
            actions = keras.layers.Input(shape=(self.action_size,), name='actions')
            
            # Add hidden layer(s) for state pathway
            net_states = layers.Dense(units=64, activation='relu')(states)
            net_states = layers.Dropout(self.dropout_rate)(net_states)
            
            net_states = layers.Dense(units=128, activation='relu')(net_states)
            net_states = layers.Dropout(self.dropout_rate)(net_states)
       
            # Add hidden layer(s) for action pathway
            net_actions = layers.Dense(units=64, activation='relu')(actions)
            net_actions = layers.Dropout(self.dropout_rate)(net_actions)
            
            net_actions = layers.Dense(units=128, activation='relu')(net_actions)
            net_actions = layers.Dropout(self.dropout_rate)(net_actions)

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
            
#            noiseStd = 0.1
#            print("critic noiseStd: ", noiseStd)
            
            # Add hidden layer(s) for state pathway
#            net_states = keras.layers.GaussianNoise(noiseStd)(states)
            net_states = layers.Dense(units=64 * bigUp, activation='relu')(states)
            net_states = layers.Dropout(self.dropout_rate)(net_states)
            
            net_states = layers.Dense(units=128 * bigUp, activation='relu')(net_states)
            net_states = layers.Dropout(self.dropout_rate)(net_states)
       
            # Add hidden layer(s) for action pathway
#            net_actions = keras.layers.GaussianNoise(noiseStd)(actions)
            net_actions = layers.Dense(units=64 * bigUp, activation='relu')(actions)
            net_actions = layers.Dropout(self.dropout_rate)(net_actions)    
        
            net_actions = layers.Dense(units=128 * bigUp, activation='relu')(net_actions)
            net_actions = layers.Dropout(self.dropout_rate)(net_actions)

            # Combine state and action pathways
            net = layers.Add()([net_states, net_actions])
            net = layers.Dense(units=128 * bigUp, activation='relu')(net)

        elif self.netArch == "QuadCopterBigNoDropout":

            # takes longer to get good results than the regular sizes copter network
            bigUp = 2 
            
            # Define input layers. The critic model needs to map (state, action) pairs to
            # their Q-values. This is reflected in the following input layers.
            states = keras.layers.Input(shape=(self.state_size,), name='states')
            actions = keras.layers.Input(shape=(self.action_size,), name='actions')
            
            # Add hidden layer(s) for state pathway
            net_states = layers.Dense(units=64 * bigUp, activation='relu')(states)            
            net_states = layers.Dense(units=128 * bigUp, activation='relu')(net_states)
       
            # Add hidden layer(s) for action pathway
            net_actions = layers.Dense(units=64 * bigUp, activation='relu')(actions)           
            net_actions = layers.Dense(units=128 * bigUp, activation='relu')(net_actions)

            # Combine state and action pathways
            net = layers.Add()([net_states, net_actions])
            net = layers.Dense(units=128 * bigUp, activation='relu')(net)

        elif self.netArch == "QuadCopterMax":

            # takes longer to get good results than the regular sizes copter network
            bigUp = 3 
            
            # Define input layers. The critic model needs to map (state, action) pairs to
            # their Q-values. This is reflected in the following input layers.
            states = keras.layers.Input(shape=(self.state_size,), name='states')
            actions = keras.layers.Input(shape=(self.action_size,), name='actions')
            
            # Add hidden layer(s) for state pathway
            net_states = layers.Dense(units=64 * bigUp, activation='relu')(states)
            net_states = layers.Dropout(self.dropout_rate)(net_states)
            
            net_states = layers.Dense(units=128 * bigUp, activation='relu')(net_states)
            net_states = layers.Dropout(self.dropout_rate)(net_states)
       
            # Add hidden layer(s) for action pathway
            net_actions = layers.Dense(units=64 * bigUp, activation='relu')(actions)
            net_actions = layers.Dropout(self.dropout_rate)(net_actions)    
        
            net_actions = layers.Dense(units=128 * bigUp, activation='relu')(net_actions)
            net_actions = layers.Dropout(self.dropout_rate)(net_actions)

            # Combine state and action pathways
            net = layers.Add()([net_states, net_actions])
            net = layers.Dense(units=128 * bigUp, activation='relu')(net)

        elif self.netArch == "QuadCopterBigELU":

            # takes longer to get good results than the regular sizes copter network
            bigUp = 2 
            
            # Define input layers. The critic model needs to map (state, action) pairs to
            # their Q-values. This is reflected in the following input layers.
            states = keras.layers.Input(shape=(self.state_size,), name='states')
            actions = keras.layers.Input(shape=(self.action_size,), name='actions')
            
            # Add hidden layer(s) for state pathway
            net_states = layers.Dense(units=64 * bigUp, activation='elu')(states)
            net_states = layers.Dense(units=128 * bigUp, activation='elu')(net_states)
       
            # Add hidden layer(s) for action pathway
            net_actions = layers.Dense(units=64 * bigUp, activation='elu')(actions)
            net_actions = layers.Dense(units=128 * bigUp, activation='elu')(net_actions)

            # Combine state and action pathways
            net = layers.Add()([net_states, net_actions])
            net = layers.Dense(units=128 * bigUp, activation='elu')(net)
   
        elif self.netArch == "QuadCopterBatchNorm":
            # Define input layers. The critic model needs to map (state, action) pairs to
            # their Q-values. This is reflected in the following input layers.
            states = keras.layers.Input(shape=(self.state_size,), name='states')
            actions = keras.layers.Input(shape=(self.action_size,), name='actions')
            
            # takes longer to get good results than the regular sizes copter network
            bigUp = 2 
            
            # Add hidden layer(s) for state pathway
            net_states = layers.Dense(units=64 * bigUp, use_bias=False)(states)
            net_states = layers.BatchNormalization()(net_states) 
            net_states = layers.Activation("relu")(net_states)
            
            net_states = layers.Dense(units=128 * bigUp, use_bias=False)(net_states)
            net_states = layers.BatchNormalization()(net_states)
            net_states = layers.Activation("relu")(net_states)
       
            # Add hidden layer(s) for action pathway
            net_actions = layers.Dense(units=64 * bigUp, use_bias=False)(actions)
            net_actions = layers.BatchNormalization()(net_actions)
            net_actions = layers.Activation("relu")(net_actions)
           
            net_actions = layers.Dense(units=128 * bigUp, use_bias=False)(net_actions)
            net_actions = layers.BatchNormalization()(net_actions)
            net_actions = layers.Activation("relu")(net_actions)

            # Combine state and action pathways
            net = layers.Add()([net_states, net_actions])
            net = layers.Dense(units=128 * bigUp, use_bias=False)(net)
            net = layers.BatchNormalization()(net) #(SMM) 
            net = layers.Activation("relu")(net)

        elif self.netArch == "QuadCopterBatchNormInput":
            # Define input layers. The critic model needs to map (state, action) pairs to
            # their Q-values. This is reflected in the following input layers.
            states = keras.layers.Input(shape=(self.state_size,), name='states')
            actions = keras.layers.Input(shape=(self.action_size,), name='actions')
            
            # takes longer to get good results than the regular sizes copter network
            bigUp = 2 
            
            # Add hidden layer(s) for state pathway
            net_states = layers.BatchNormalization()(states)
            net_states = layers.Dense(units=64 * bigUp, activation='relu')(net_states)
            net_states = layers.Dropout(self.dropout_rate)(net_states)
            
            net_states = layers.Dense(units=128 * bigUp, activation='relu')(net_states)
            net_states = layers.Dropout(self.dropout_rate)(net_states)
       
            # Add hidden layer(s) for action pathway
            net_actions = layers.BatchNormalization()(actions)
            net_actions = layers.Dense(units=64 * bigUp, activation='relu')(actions)
            net_actions = layers.Dropout(self.dropout_rate)(net_actions)    
        
            net_actions = layers.Dense(units=128 * bigUp, activation='relu')(net_actions)
            net_actions = layers.Dropout(self.dropout_rate)(net_actions)

            # Combine state and action pathways
            net = layers.Add()([net_states, net_actions])
            net = layers.Dense(units=128 * bigUp, activation='relu')(net)
    
        elif self.netArch == "imageInputV1":     
            # for img state space
            actions = keras.layers.Input(shape=(self.action_size,), name='actions')
            # Add hidden layer(s) for action pathway
            net_actions = keras.layers.Dense(units=512, activation='relu')(actions)
    
            states = keras.layers.Input(shape=(96, 96, 3), name='states')
            net_states = keras.layers.Conv2D(32, (8, 8), strides=[4, 4], padding='same', activation='relu')(states)
            net_states = keras.layers.MaxPooling2D(pool_size=2)(net_states)
            net_states = keras.layers.Dropout(self.dropout_rate)(net_states)
            net_states = keras.layers.Conv2D(64, (4, 4), strides=[2, 2], padding='same', activation='relu')(net_states)
            net_states = keras.layers.MaxPooling2D(pool_size=2)(net_states)
            net_states = keras.layers.Dropout(self.dropout_rate)(net_states)
            net_states = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(net_states)
            net_states = keras.layers.MaxPooling2D(pool_size=2)(net_states)
            net_states = keras.layers.Dropout(self.dropout_rate)(net_states)
            net_states = keras.layers.Flatten()(net_states)
            net_states = keras.layers.Dense(units=512, activation='relu')(net_states)
            net_states = keras.layers.Dropout(self.dropout_rate)(net_states)
    
            net = keras.layers.Add()([net_states, net_actions])
    
            # Add more layers to the combined network if needed
            net = keras.layers.Dense(units=256, activation='relu')(net)
            net = keras.layers.Dropout(self.dropout_rate)(net)
             

        # Add final output layer to produce action values (Q values). The final output
        # of this model is the Q-value for any given (state, action) pair.
        Q_values = keras.layers.Dense(units=1, activation=None, name='q_values')(net)

        # Create Keras model
        self.model = keras.models.Model(inputs=[states, actions], outputs=Q_values)
        # Define optimizer and compile model for training with built-in loss function
        optimizer = keras.optimizers.Adam(lr=self.learning_rate)
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

def descale_output(x, action_range, action_low):
    temp = (np.array(x) / np.array(action_range)) - np.array(action_low)
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

        print("*************************************")
        print("*************************************")
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

        """
        --- Experiment Notes ---
        
        solve mt climber with copter network
            1. 75 episodes (unstable solution) with batch size/buffer 32/1024, gamma/tau .99/.01
               learning rate 0.0001, explore decay 0.00001, action repeat 1
            2. 140 episodes (unstable solution) with batch size/buffer 128/1024, gamma/tau .995/.005
               learning rate 0.0001, explore decay 0.00001, action repeat 1
            3. 90 episodes (stable solution) with batch size/buffer 256/10000, gamma/tau .995/.005
               learning rate 0.0001, explore decay 0.00001, action repeat 1
            4. ~170 episodes (no solution) with batch size/buffer 512/10000, gamma/tau .995/.005
               learning rate 0.0001, explore decay 0.00001, action repeat 1, learn freq 2
               
        solve mt climber with copter BIG network
            1. 100 episodes (stable solution) with batch size/buffer 256/10,000, gamma/tau .995/.005
               learning rate 0.0001, explore decay 0.00001, action repeat 1
            2. ~140 episodes (no solution) with batch size/buffer 256/10,000, gamma/tau 0.0/1.0
               learning rate 0.0001, explore decay 0.00001, action repeat 1
            3. 50 episodes (stable solution) with batch size/buffer 256/10,000, gamma/tau .995/1.0
               learning rate 0.0001, explore decay 0.00001, action repeat 1, *no soft update*
            4. ~700 episodes (converging, but no solution) with batch size/buffer 256/10000, gamma/tau .995/1.0
               learning rate 0.0001, explore decay 0.00001, action repeat 10, *no soft update*
               perhaps the explore rate is too low for the number of episodes?
            5. ~204 episodes (no solution) with batch size/buffer 256/10,000, gamma/tau .9/1.0
               learning rate 0.0001, explore decay 0.00001, action repeat 1, *no soft update*
            
            6. 60 episodes (stable solution) with batch size/buffer 128/10,000, gamma/tau .995/.005
               learning rate 0.0001, explore decay 0.00001, action repeat 1, with soft update
            7. 110 episodes (stable solution) with batch size/buffer 128/100,000, gamma/tau .995/.005
               learning rate 0.0001, explore decay 0.00001, action repeat 1, with soft update
            8. 82 episodes (stable solution) with batch size/buffer 128/100,000, gamma/tau .995/.005
               learning rate 0.0001, explore decay 0.00001, action repeat 1, with soft update, 
               *remove L2 norm of actor network - seems to help*
            
            9. 2x experiments, 104/115  episodes (stable solution) with batch size/buffer 128/100,000, gamma/tau .995/.005
               learning rate 0.0001, explore decay 0.00001, action repeat 1, with soft update, 
            10. 2x experiments, ~150 episodes (no solution) with batch size/buffer 128/100,000, gamma/tau .995/.005
               learning rate 0.0001, explore decay 0.0001, action repeat 1, with soft update, 
            11. 2x experiments, ~150 episodes (no solution) with batch size/buffer 256/100,000, gamma/tau .995/.005
               learning rate 0.00025, explore decay 0.0001, action repeat 1, with soft update, 
            12. 2x experiments, 160/310 episodes (stable solution) with batch size/buffer 128/100,000, gamma/tau .99/.01
               learning rate 0.00001, explore decay 0.00001, action repeat 1, with soft update,  adam with amsgrad
            13. 2x experiments, 40/70 episodes (stable solution) with batch size/buffer 128/100,000, gamma/tau .99/.01
               learning rate 0.0001, explore decay 0.00001, action repeat 1, with soft update, 0.1 Dropout
            
            14. 200 episodes (no solution) with batch size/buffer 128/100,000, gamma/tau .995/.005
               learning rate 0.0001, explore decay 0.00001, action repeat 1, with soft update, 
               *remove L2 norm of actor network*, 128 instead of 64 final crtiic hidden layer plus NO dropout
            15. ~251 episodes (no solution) with batch size/buffer 256/10000, gamma/tau .995/.005
               learning rate 0.0001, explore decay 0.00001, action repeat 1, learn freq 10, batch norm
            16. 2x experiments, 40/95 episodes (stable solution) with batch size/buffer 128/100,000, gamma/tau .99/.01
               learning rate 0.0001, explore decay 0.00001, action repeat 1, with soft update, NO DROPOUT

            17. 1x experiments, 250 episodes (no solution) with batch size/buffer 128/100,000, gamma/tau .99/.01,
               learning rate 0.0001, explore decay 0.00001, action repeat 1, with soft update, NO DROPOUT, 0.001 L2 on actor/critic
            18. 1x experiments, 50 episodes (stable solution) with batch size/buffer 128/100,000, gamma/tau .99/.01,
               learning rate 0.0001, explore decay 0.00001, action repeat 1, with soft update, NO DROPOUT, NO 0.001 L2 on actor/critic
            19. 1x experiments, 65 episodes (stable solution) with batch size/buffer 128/100,000, gamma/tau .99/.01,
               learning rate 0.0001, explore decay 0.00001, action repeat 1, with soft update, NO DROPOUT, NO 0.001 L2 on actor/critic

            20. 1x experiments, ~150 episodes (no solution) with batch size/buffer 128/1,000,000, gamma/tau .99/.01,
               learning rate 0.0001, explore decay 0.00001, action repeat 1, with soft update, NO DROPOUT, NO 0.001 L2 on actor/critic
            21. 1x experiments, 85 episodes (stable solution) with batch size/buffer 128/1,000,000, gamma/tau .99/.01,
               learning rate 0.0001, explore decay 0.00001, action repeat 1, with soft update, 0.2 DROPOUT, NO 0.001 L2 on actor/critic

            22. 1x experiments, 20 episodes (stable solution) with batch size/buffer 128/1,000,000, gamma/tau .99/.01,
               learning rate 0.001, explore decay 0.00001, action repeat 1, with soft update, 0.2 DROPOUT, NO 0.001 L2 on actor/critic
               no explore after initial random buffer filling

        solve mt climber with copter MAX network
            1. 323 episodes (no solution) with batch size/buffer 128/100,000, gamma/tau .995/.005
               learning rate 0.0001, explore decay 0.00001, action repeat 1, NO soft update, 
               *remove L2 norm of actor network*, 128 instead of 64 final crtiic hidden layer 
            2. 2x experiments, 125/? episodes (stable solution) with batch size/buffer 64/100,000, gamma/tau .995/.005
               learning rate 0.0001, explore decay 0.00001, action repeat 1, NO soft update (almost twice as fast), 
               small batch seems to help convergence (double batch size is like doubling the learning rate?)
               bigup 3 (big network size, GPU seems fine)
            3. 60 episodes (stable solution) with batch size/buffer 128/100,000, gamma/tau .99/.01
               learning rate 0.0001, explore decay 0.00001, action repeat 1, with soft update, 0.1 dropout
            
            4. experiments, 140 episodes (stable solution) with batch size/buffer 64/100,000, gamma/tau .99/.01
               learning rate 0.0001, explore decay 0.00001, action repeat 1, with soft update, no dropout
            5. experiments, 220 episodes (stable solution) with batch size/buffer 64/100,000, gamma/tau .99/.01
               learning rate 0.0001, explore decay 0.00001, action repeat 1, with soft update, 0.3 dropout
            6. experiments, ~310 episodes (no solution) with batch size/buffer 64/100,000, gamma/tau .99/.01
               learning rate 0.0001, explore decay 0.00001, action repeat 1, with soft update, 0.5 dropout
            7. experiments, ~310 episodes (no solution) with batch size/buffer 64/100,000, gamma/tau .99/.01
               learning rate 0.0001, explore decay 0.00001, action repeat 1, with soft update, 0.7 dropout
               

        solve mt climber with copter BIG network ELU activation  
            1. solve once in 10 episodes, but other run no solution, with batch size/buffer 128/100,000, gamma/tau .995/.005
               learning rate 0.0001, explore decay 0.00001, action repeat 1, with soft update, 
               *remove L2 norm of actor network*, 128 instead of 64 final crtici hidden layer plus NO dropout
            2. 2x experiments, 250 episodes (no solution) with batch size/buffer 128/100,000, gamma/tau .99/.01
               learning rate 0.0001, explore decay 0.00001, action repeat 1, with soft update, NO DROPOUT

       
        solve mt climber with Lillicrap network


        solve mt climber with uniform layers network
            2*100 neurons
            1. 1x experiments, 300 episodes (no solution) with batch size/buffer 128/100,000, gamma/tau .99/.01
               learning rate 0.0001, explore decay 0.00001, action repeat 1, with soft update, NO DROPOUT, no Xavier init
            
            2*200 neurons
            1. 1x experiments, 300 episodes (no solution) with batch size/buffer 128/100,000, gamma/tau .99/.01
               learning rate 0.0001, explore decay 0.00001, action repeat 1, with soft update, NO DROPOUT, no Xavier init

            5*300 neurons
            1. 1x experiments, 300 episodes (unstable solution) with batch size/buffer 128/100,000, gamma/tau .99/.01
               learning rate 0.0001, explore decay 0.00001, action repeat 1, with soft update, NO DROPOUT, no Xavier init

        
            2*500 neurons
            1. 2x experiments, 250 episodes (no solution) with batch size/buffer 128/100,000, gamma/tau .99/.01
               learning rate 0.0001, explore decay 0.00001, action repeat 1, with soft update, NO DROPOUT, Xavier init
            1. 1x experiments, 250 episodes (no solution) with batch size/buffer 128/100,000, gamma/tau .99/.01
               learning rate 0.0001, explore decay 0.00001, action repeat 1, with soft update, NO DROPOUT, no Xavier init
               


        try sinusoidal explore rates

          

        Is action repeat useful?
        
        It is interesting in that often the agent seems to get worse actions before getting better, similar
        to language aquisition among humans stated in "The Learning Brain" Great Courses by Prof. Thad A. Polk.
        Also from this course are the priciples for learning of randomization, spacing, and challenge.
        
        Too large batch sizes (>256) seem to lead to overlearning and gradient decent not converging
        
        Learning frequency seems to be roughly linearly correlated with how many episodes to solve. Still, this
        approach is still able to get a solution eventually, learning one to believe that one does not have 
        to train every episode to find a solution.

        """

        # Action Repeat
        self.action_repeat = 2
        self.state_size = env.observation_space.shape[0] * self.action_repeat

        # select network based on enviromnet type
        self.learningRate = 0.00001  # 0.0001 default MtC, 0.00025 Atari paper learning rate, 0.0000625 Rainbow learning rate
        self.learnFrequency = 1 # how many steps per training
        self.dropoutRate = 0.2
        # QuadCopter, QuadCopterBig, QuadCopterMax, QuadCopterBigELU, QuadCopterBigNoDropout, QuadCopterBatchNorm
        # Lillicrap, Hausknecht
        network_arch = "QuadCopterBig"
        if envType == "imageStateContinuousAction":
            network_arch = "imageInputV1"

        # Actor (Policy) Model
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high, network_arch, \
                                 self.learningRate, self.dropoutRate)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high, network_arch, \
                                  self.learningRate, self.dropoutRate)

        # Critic (Value) Model
        self.critic_local = Critic(self.state_size, self.action_size, network_arch, self.learningRate, self.dropoutRate)
        self.critic_target = Critic(self.state_size, self.action_size, network_arch, self.learningRate, self.dropoutRate)

        # Initialize target model parameters with local model parameters
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())

        # Replay memory
        if self.envType == "continousStateAction":
            self.buffer_size = 1000000 # 1,000,000 is standard. Most episodes are around 1000 steps in OpenAI for a complete run 
            self.batch_size = 256 # 128 for copter big gives good results
        elif self.envType == "imageStateContinuousAction":
            self.buffer_size = 10000 # 100000 in other solution to car racing with DDQN with dropout
            self.batch_size = 32
        else:    
            raise("\nDDPG:__init__: ERROR! unsupported env type!\n")            
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

        # Discount factor (percentage to use) for Q_targets_next from the critic model added to the rewards for training
        self.gamma = 0.99 #0.99  #0.995

        # Actor-Critic local and target soft update ratio of target parameters, off-policy learning algorithm
        self.useSoftUpdates = True
        self.tau = 0.01 # 1.0 # 0.005 #0.01, percentage of local weights to put into the target network
    
        # Exploration Policy (expodential decay based on lifetime steps)
        self.explore_start = 1.0            # exploration probability at start
        self.explore_stop = 0.00            # minimum exploration probability 
        self.explore_decay_rate = 0.95       # amount of explore prob to keep each episode
        self.exploreStep = 0
        self.explore_p = self.explore_start

        # step and episode counters   
        self.stepCount = 0
        self.i_episode = 0

        # Save the Q targets and action gradients for analysis and avoid memory stack allocations
        self.Q_targets = 0
        self.action_gradients = 0

        # print out the network paramters for experimental result logging
        print("*************************************")
        print("*** DDPG Agent Paramter ***")
        print("- network architecture chosen: ", network_arch)
        print("[ ACTOR MODEL SUMMARY ]")
        self.actor_local.model.summary()
        print("[ CRITIC MODEL SUMMARY ]")
        self.critic_local.model.summary()
        print("- action_repeat: ", self.action_repeat)
        print("- learningRate: ", self.learningRate)
        print("- learnFrequency: ", self.learnFrequency)
        print("- dropout_rate: ", self.dropoutRate)
        print("- buffer_size: ", self.buffer_size)
        print("- batch_size: ", self.batch_size)
        print("- gamma: ", self.gamma)
        print("- useSoftUpdates: ", self.useSoftUpdates)
        print("- tau: ", self.tau)
        print("- explore_start: ", self.explore_start)
        print("- explore_stop: ", self.explore_stop)
        print("- explore_decay_rate: ", self.explore_decay_rate)
        print("*************************************")

    def reset_episode(self, state):
        
        # expand the state returned from the gym environment according action_repeat.
        state = np.concatenate([state] * self.action_repeat) 
        
        if self.envType == "imageStateContinuousAction":
            state = unit_image(state) # normalize to between 0-1
        
        self.last_state = state
        
        # increase episode counters
        self.i_episode += 1
        
        if len(self.memory) > self.memory_full_enough(): # batch_size, fill buffer_size before training to stabilize training process??
            self.exploreStep += 1
            self.explore_p *= self.explore_decay_rate 
            
            # cycle the exploration policy between explore and exploit
            # this helps the agent get unstuck follow bad actions over and over
            # additonally, without training, there are some state-action pairs that will never be available to explore
#            if self.explore_p < self.explore_stop * 2:
#                self.exploreStep = self.explore_start           

        print("\tresetting episode... next explore_p: ", self.explore_p)
        
        return state

    def memory_full_enough(self):
        return self.batch_size * 10 * self.action_size * self.action_size

    def step(self, next_state):
        
        # Ensure that size of next_state as returned from the 
        # environment is increased in according to the action_repeat
        next_state = np.concatenate([next_state] * self.action_repeat) 
        
        if self.envType == "imageStateContinuousAction":
            next_state = unit_image(next_state)
        
        # increase step count
        self.stepCount += 1

        # Roll over last state and action
        self.last_state = next_state

                
    def act(self, state, mode="train"):       
                 
        # Ensure that size of next_state as returned from the 
        # environment is increased in according to the action_repeat
        state = np.concatenate([state] * self.action_repeat) 
        
        if self.envType == "imageStateContinuousAction":
            state = unit_image(state)
       
        # return a random action if memory is not filled (inital network weights)
        if len(self.memory) < self.memory_full_enough(): # batch_size, fill buffer_size before training to stabilize training process??
            action = self.env.action_space.sample()
             # for car racing env, choose the max of gas or brake
            if self.envType == "imageStateContinuousAction":
                if action[1] > action [2]:
                    action[2] = 0
                else:
                    action[1] = 0

            return action
        
        # Explore or Exploit
        # Use expodentially decaying noise, more consistant results across environments than OU noise
        # use sinusoid instead, up and down?
        if self.explore_p > np.random.rand() and mode == "train":
            # Make a random action if in training mode to explore the environment
#            action = self.env.action_space.sample()       
            
            # use correlated noise, with a percentage of noise based on the current explore rate
            if self.envType == "continousStateAction":
                state = np.reshape(state, [-1, self.state_size])
            elif self.envType == "imageStateContinuousAction":
                state = np.expand_dims(state, axis=0)  # for img state space

            agentAction = self.actor_local.model.predict(state)[0]                 
            randAction = self.env.action_space.sample()
            action = agentAction * (1-self.explore_p) + randAction * self.explore_p            
                       
        else:
            """Returns action(s) for given state(s) as per current policy."""
            if self.envType == "continousStateAction":
                state = np.reshape(state, [-1, self.state_size])
            elif self.envType == "imageStateContinuousAction":
                state = np.expand_dims(state, axis=0)  # for img state space

            action = self.actor_local.model.predict(state)[0]  
            action = scale_output(action, self.action_range, self.action_low)

            # Making the actions partially random seems to hurt the agent in finding the right actions
            # probably the correlation between the agent weights and state and actions gets
            # messed up and is not consistant
            # make an agent action proportional to 1 - explore_p
#            agentAction = self.actor_local.model.predict(state)[0]                 
#            randAction = self.env.action_space.sample()
#            action = agentAction * (1-self.explore_p) + randAction * self.explore_p            
#            print("\tagent steps cnt: ", self.stepCount, ", with action:", action)
            

        # for car racing env, choose the max of gas or brake
        if self.envType == "imageStateContinuousAction":
            if action[1] > action [2]:
                action[2] = 0
            else:
                action[1] = 0
               
        return action

    def learn(self, action, reward, next_state, done):
        
        # Ensure that size of next_state as returned from the 
        # environment is increased in size according to action_repeat
        next_state = np.concatenate([next_state] * self.action_repeat) 
        
        if self.envType == "imageStateContinuousAction":
            next_state = unit_image(next_state)
        
        # Save experience / reward
        self.memory.add(self.last_state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        # Check step count to avoid loading and unloading the GPU all the time
        if len(self.memory) > self.memory_full_enough() and self.stepCount % self.learnFrequency == 0:
#            print("\t\tlearning on total training step count: ", self.stepCount)
            experiences = self.memory.sample(self.batch_size)
        
            """Update policy and value parameters using given batch of experience tuples."""
            # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
            actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
            rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
            dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
   
    	    # turn the states and next_states into numpy arrays, this is 
            # important change from copter DDPG in order to to properly stack
            # image states, as vstack won't work properly on multiple dimensions	
#            states = np.vstack([e.state for e in experiences if e is not None])
            states = []
            for e in experiences:
                states.append(e.state)
    
            states = np.array(states)

#            next_states = np.vstack([e.next_state for e in experiences if e is not None])
            next_states = []
            for e in experiences:
                next_states.append(e.next_state)
    
            next_states = np.array(next_states)
    
            if self.useSoftUpdates:       
                # Get predicted next-state actions and Q values from target models    
                actions_next = self.actor_target.model.predict_on_batch(next_states)
                for a in actions_next: # scale output to match env ranges
                    a = scale_output(a, self.action_range, self.action_low)
                    if self.envType == "imageStateContinuousAction":
                        if a[1] > a[2]:
                            a[2] = 0
                        else:
                            a[1] = 0

                Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])
        
                # Compute Q targets for current states and train critic model (local)
                self.Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
                self.critic_local.model.train_on_batch(x=[states, actions], y=self.Q_targets)
            else: 
                # Get predicted next-state actions and Q values from local models    
                actions_next = self.actor_local.model.predict_on_batch(next_states)
                for a in actions_next: # scale output to match env ranges
                    a = scale_output(a, self.action_range, self.action_low)

                Q_targets_next = self.critic_local.model.predict_on_batch([next_states, actions_next])
        
                # Compute Q targets for current states and train critic model (local)
                self.Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
                self.critic_local.model.train_on_batch(x=[states, actions], y=self.Q_targets)
    
            # Train actor model (local)
            self.action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
            self.actor_local.train_fn([states, self.action_gradients, 1])  # custom training function
    
            # Soft-update target models
            if self.useSoftUpdates:       
                self.soft_update(self.critic_local.model, self.critic_target.model)
                self.soft_update(self.actor_local.model, self.actor_target.model)   

    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        assert len(local_weights) == len(target_weights), "Local and target model parameters must have the same size"

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)   
