import numpy

from marioagent import MarioAgent
import pdb
import pygame
import numpy as np
import argparse
from collections import deque
from config import Config
from model import Model


class LearningAgent(MarioAgent):
    """ In fact the Python twin of the
        corresponding Java ForwardAgent.
    """
        

    def __init__(self, dim_obs, dim_action, model_name):
        self.config = Config()
        self.obs = None
        self.action_dim = dim_action
        self.model = Model(dim_obs,dim_action, self.config, model_name, demo_mode=False)
        self.eps = self.config.INITIAL_EPS
        self.burn_in_size = self.config.BURN_IN_SIZE
        self.inv_gamma = 1.0 / self.config.GAMMA
        self.n_pow_gamma = self.config.GAMMA ** self.config.N_STEP
        self.step = 0
        self.burn_in_cur_size = 0
        self.last_obs = None
    
        """Constructor"""

    def reset(self):
        self.isEpisodeOver = 0
        self.obs = None
        self.last_obs = None


    def integrateObservation(self, obs):
        self.obs = np.array(obs)

    def getAction(self):
        if self.burn_in_cur_size < self.burn_in_size:
            action = np.random.randint(self.action_dim)
        else:
            action = self.epsilon_greedy_action(self.obs)
        return action

    def update_network(self, cur_obs, action, reward, next_obs, pretrain=False):
        # deal with episode end
        done = reward is None
        self.perceive(cur_obs, action, reward, next_obs, done)
        if self.burn_in_cur_size < self.burn_in_size:
            self.burn_in_cur_size += 1
            if self.burn_in_cur_size % 1000 == 0:
                print(self.burn_in_cur_size)
            return
        self.model.train(self.step, pretrain=pretrain)
        #self.model.update_target(self.step)
        self.update_eps()
        self.step += 1

    def epsilon_greedy_action(self, state):
        p = np.random.random_sample()
        q_values = self.model.predict(state)
        if p <= self.eps:
            action = np.random.randint(self.action_dim)
        else:
            action = np.argmax(q_values, axis=1)[0]
        return action

    def update_eps(self):
        delta = (self.config.INITIAL_EPS - self.config.FINAL_EPS) / 100000
        if self.eps > self.config.FINAL_EPS:
            self.eps -= delta

    def perceive(self, cur_obs, action, reward, next_obs, done):
        """
            Append the transition to the replay buffer
        """
        #s_t0, a_t0, r_t1, s_t1, d_t0
        if self.last_obs is not None:
            if reward is None:
                self.last_obs[4] = True
                self.model.perceive(self.last_obs)
            else:
                self.model.perceive(self.last_obs)
                self.last_obs = [cur_obs, action, reward, next_obs, done]
        else:
            self.last_obs = [cur_obs, action, reward, next_obs, done]
        
        
    def cal_discount_reward(self, reward_queue):
        weight = 1
        ret = 0.0
        for r in reward_queue:
            ret += weight * r
            weight *= self.config.GAMMA
        return ret
