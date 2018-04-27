import numpy

from marioagent import MarioAgent
import pdb
import pygame
import numpy as np
import argparse
from collections import deque
from config import Config
from a2c import A2C

class PGAgent(MarioAgent):

    def __init__(self, dim_obs, dim_action, model_config, lr, critic_lr, n):
        self.n = n
        self.model = A2C(model_config, lr, critic_lr, n)
    
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

    def update_network(self, states, actions, rewards):
        self.model.train(states, actions, rewards)

    def cal_discount_reward(self, reward_queue):
        weight = 1
        ret = 0.0
        for r in reward_queue:
            ret += weight * r
            weight *= self.config.GAMMA
        return ret
