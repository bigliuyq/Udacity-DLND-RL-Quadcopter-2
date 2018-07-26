import numpy as np

import random
from collections import namedtuple, deque

import os
import pandas as pd
from agents import util


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu=None, theta=0.15, sigma=0.3):
        """Initialize parameters and noise process."""
        self.size = size
        self.mu = mu if mu is not None else np.zeros(self.size)
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.size) * self.mu
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = self.mu

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state        


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
                
        
from keras import layers, models, optimizers, regularizers,initializers
from keras import backend as K


class Actor:
    """Actor (Policy) Model."""

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

        # Initialize any other variables here

        self.build_model()

    def build_model(self):
        """Build an actor (policy) network that maps states -> actions."""
        # Define input layer (states)   输入层，并做normalization处理
        states = layers.Input(shape=(self.state_size,), name='states')
        layers.normalization.BatchNormalization()                

        # Add hidden layers  隐含层分别采用400、300、200单元数，使用relu激活，并用dropout防止过拟合
        net = layers.Dense(units=400)(states)
        #net=layers.BatchNormalization()(net)
        net=layers.Activation('relu')(net)
        net=layers.Dropout(0.2)(net)
        
        net = layers.Dense(units=300)(net)
        #net=layers.BatchNormalization()(net)
        net=layers.Activation('relu')(net)
        net=layers.Dropout(0.2)(net)
        
        net = layers.Dense(units=200)(net)
        #net=layers.BatchNormalization()(net)
        net=layers.Activation('relu')(net)
        net=layers.Dropout(0.2)(net)
        
        # Add final output layer with sigmoid activation  输出层使用权重初始化，给予较小初始权重，使用sigmoid激活函数
        raw_actions = layers.Dense(units=self.action_size,
                kernel_initializer=initializers.RandomUniform(minval=-0.0003, maxval=0.0003),activity_regularizer=regularizers.l2(0.0001),name='raw_actions')(net)
        #raw_actions=layers.BatchNormalization()(raw_actions)
        raw_actions=layers.Activation('sigmoid')(raw_actions)

        # Scale [0, 1] output for each action dimension to proper range
        actions = layers.Lambda(lambda x: (x * self.action_range) + self.action_low,
            name='actions')(raw_actions)

        # Create Keras model
        self.model = models.Model(inputs=states, outputs=actions)

        # Define loss function using action value (Q value) gradients
        action_gradients = layers.Input(shape=(self.action_size,))
        loss = K.mean(-action_gradients * actions)

        # Define optimizer and training function
        optimizer = optimizers.Adam(lr=0.0003)
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = K.function(
            inputs=[self.model.input, action_gradients, K.learning_phase()],
            outputs=[],
            updates=updates_op)        
        
        
class Critic:
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size

        # Initialize any other variables here

        self.build_model()

    def build_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # Define input layers
        states = layers.Input(shape=(self.state_size,), name='states')
        layers.normalization.BatchNormalization()        
        actions = layers.Input(shape=(self.action_size,), name='actions')
        #layers.normalization.BatchNormalization()

        # Add hidden layer(s) for state pathway
        net_states = layers.Dense(units=400)(states)       
        #net_states=layers.BatchNormalization()(net_states)
        net_states=layers.Activation('relu')(net_states)
        net_states=layers.Dropout(0.2)(net_states)
        net_states = layers.Dense(units=300)(net_states)
        #net_states=layers.BatchNormalization()(net_states)
        net_states=layers.Activation('relu')(net_states)
        net_states=layers.Dropout(0.2)(net_states)

        # Add hidden layer(s) for action pathway
        net_actions = layers.Dense(units=400)(actions)
        #net_actions=layers.BatchNormalization()(net_actions)
        net_actions=layers.Activation('relu')(net_actions)
        net_actions=layers.Dropout(0.2)(net_actions)
        net_actions = layers.Dense(units=300)(net_actions)
        #net_actions=layers.BatchNormalization()(net_actions)
        net_actions=layers.Activation('relu')(net_actions)        
        net_actions=layers.Dropout(0.2)(net_actions)

        #activity_regularizer=regularizers.l2(0.0001)
        #kernel_initializer=initializers.RandomUniform(minval=-0.0577,maxval=0.0577)

        # Combine state and action pathways
        net = layers.Add()([net_states, net_actions])
        #net = layers.Concatenate()([net_states, net_actions])
        #net=layers.BatchNormalization()(net)
        net = layers.Activation('relu')(net)
        #net=layers.Dropout(0.1)(net)        

        # Add final output layer to prduce action values (Q values)
        Q_values = layers.Dense(units=1,activity_regularizer=regularizers.l2(0.0001),name='q_values')(net)
        #Q_values=layers.BatchNormalization()(Q_values)
        #Q_values = layers.Activation('tanh')(Q_values)

        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        # Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam(lr=0.001)
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = K.gradients(Q_values, actions)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)        


        
class DDPG():
    """Reinforcement Learning agent that learns using DDPG."""
    def __init__(self, task):
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high

        # Actor (Policy) Model
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high)

        # Critic (Value) Model
        self.critic_local = Critic(self.state_size, self.action_size)
        self.critic_target = Critic(self.state_size, self.action_size)

        # Initialize target model parameters with local model parameters
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())

        # Noise process
        self.exploration_mu = None
        self.exploration_theta = 0.15
        self.exploration_sigma = 0.2
        self.noise = OUNoise(self.action_size, self.exploration_mu, self.exploration_theta, self.exploration_sigma)

        # Replay memory
        self.buffer_size = 100000
        self.batch_size = 64
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

        # Algorithm parameters
        self.gamma = 0.99  # discount factor
        self.tau = 0.0001  # for soft update of target parameters
                
        self.best_score = -np.inf
        
        # Save episode stats
        self.stats_filename = os.path.join(
            'out',
            "stats_{}.csv".format(util.get_timestamp()))  # path to CSV file
        self.stats_columns = ['episode', 'total_reward']  # specify columns to save
        self.episode_num = 1
        print("Saving stats {} to {}".format(self.stats_columns, self.stats_filename))  # [debug]
        
        
    #定义一个辅助函数来写入 CSV 文件
    def write_stats(self, stats):
        """Write single episode stats to CSV file."""
        df_stats = pd.DataFrame([stats], columns=self.stats_columns)  # single-row dataframe
        df_stats.to_csv(self.stats_filename, mode='a', index=False,
            header=not os.path.isfile(self.stats_filename))  # write header first time only
                
        
    def reset_episode(self):
        self.total_reward = 0.0
        self.count = 0
        self.noise.reset()
        state = self.task.reset()
        self.last_state = state
        #self.last_action = None
        return state

    def step(self, action, reward, next_state, done):
         # Save experience / reward
        self.total_reward += reward
        self.count += 1
        self.memory.add(self.last_state, action, reward, next_state, done)
        
        if done:
            self.score = self.total_reward / float(self.count) if self.count else 0.0
            if self.score > self.best_score:
                self.best_score = self.score
            # Write episode stats
            self.write_stats([self.episode_num, self.total_reward])
            self.episode_num += 1

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

        # Roll over last state and action
        self.last_state = next_state

    def act(self, states):
        """Returns actions for given state(s) as per current policy."""
        states = np.reshape(states, [-1, self.state_size])
        action = self.actor_local.model.predict(states)[0]
        return list(action + self.noise.sample())  # add some noise for exploration

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples."""
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        #rewards=np.tanh(rewards)
        
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])

        # Get predicted next-state actions and Q values from target models
        #     Q_targets_next = critic_target(next_state, actor_target(next_state))
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
        
        
        