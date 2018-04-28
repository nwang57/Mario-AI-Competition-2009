from agents.marioagent import MarioAgent
import pdb
import numpy as np
import argparse
from collections import deque
from a2c import A2C

class PGAgent(MarioAgent):

    def __init__(self, dim_obs, dim_action, model_config_path, lr, critic_lr, n):
        self.n = n
        self.model = A2C(model_config_path, lr, critic_lr, n)
    
        """Constructor"""

    def reset(self):
        self.isEpisodeOver = 0
        self.obs = None

    def integrateObservation(self, obs):
        self.obs = obs

    def getAction(self):
        return A2C.get_action(self.model.model, self.obs)

    def update_network(self, n_ep, states, actions, rewards, gamma):
        self.model.train(n_ep, states, actions, rewards, gamma)
