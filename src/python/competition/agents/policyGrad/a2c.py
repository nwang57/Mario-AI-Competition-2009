import sys
import argparse
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Lambda, Conv2D, Flatten
from keras.models import load_model
from keras import optimizers

from reinforce import Reinforce

import functools

def lazy_property(func):
    attribute = '_lazy_' + func.__name__

    @property
    @functools.wraps(func)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, func(self))
        return getattr(self, attribute)
    return wrapper

class A2C(Reinforce):
    # Implementation of N-step Advantage Actor Critic.
    # This class inherits the Reinforce class, so for example, you can reuse
    # generate_episode() here.

    def __init__(self, model_path, lr, critic_lr, n=20,
        output_file="model", weight_file=None, critic_weight_file=None):
        with open(model_path, 'r') as f:
            model = keras.models.model_from_json(f.read())
        super(A2C, self).__init__(model, lr, output_file, weight_file)

        # Initializes A2C.
        # Args:
        # - model: The actor model.
        # - lr: Learning rate for the actor model.
        # - critic_model: The critic model.
        # - critic_lr: Learning rate for the critic model.
        # - n: The value of N in N-step A2C.
        self.model.summary()
        self.input_dim = int(self.input_dim)
        self.output_dim = 1
        self.critic_model = self.build_critic(critic_lr)
        self.critic_model.summary()
        self.n = n
        self.eval_freq = 200
        self.actor_update_freq = 5

        self.critic_weight_file = critic_weight_file
        if self.critic_weight_file is not None:
            self.critic_model.load_weights(self.critic_weight_file)

        self.loss1 = -tf.reduce_mean(tf.multiply(self.G, tf.log(tf.reduce_sum(tf.multiply(self.A, self.output), axis=1, keepdims=True))))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.opt = self.optimizer.minimize(self.loss1)
        self.grad = self.optimizer.compute_gradients(self.loss1)

    def save_weights(self, n_ep):
        super(A2C, self).save_weights(n_ep)
        self.critic_model.save_weights("%s_critic_%d.h5" % (self.output_file, n_ep))

    def load_critic_weights(self):
        self.model.load_weights(self.critic_weight_file)

    def build_critic(self, lr):
        model = Sequential()
        model.add(Dense(32, input_dim=self.input_dim, activation='relu', use_bias=True))
        model.add(Dense(32, activation='relu', use_bias=True))
        model.add(Dense(32, activation='relu', use_bias=True))
        model.add(Dense(self.output_dim, activation='linear', kernel_initializer='uniform'))
        opt = optimizers.Adam(lr = lr)
        model.compile(optimizer=opt, loss='mean_squared_error')
        return model

    def train(self, n_ep, states, actions, rewards, gamma=0.99):
        # Trains the model on a single episode using A2C.
        gamma_n = gamma ** self.n

        rewards *= 0.01
        values = self.critic_model.predict(states).flatten()
        T = len(rewards)
        rewards = np.concatenate((rewards, [0]*(self.n-1)))
        values = np.concatenate((values, [0]*(self.n-1)))

        R = np.zeros((1,T))
        tmp = [0]*T
        R[0, T-1] = rewards[T-1]
        tmp[T-1] = rewards[T-1]
        for i in range(T-2,-1,-1):
            tmp[i] = tmp[i+1]*gamma + rewards[i] - rewards[i+self.n]*gamma_n
            R[0, i] = tmp[i] + values[i+self.n]*gamma_n
        G = R - values[:T]
        if n_ep % self.actor_update_freq == 0:
            _, o, loss = self.sess.run([self.opt, self.output, self.loss1], feed_dict = {self.A: actions, self.input: states, self.G: np.transpose(G)})
        self.critic_model.fit(states, R.T, verbose = 0)
