import numpy
__author__ = "Sergey Karakovskiy, sergey at idsia fullstop ch"
__date__ = "$May 1, 2009 2:46:34 AM$"

from marioagent import MarioAgent
import pdb
import pygame

class HumanAgent(MarioAgent):
    """ In fact the Python twin of the
        corresponding Java ForwardAgent.
    """

    KEY_JUMP = 3
    KEY_SPEED = 4

    def reset(self):
        self.isEpisodeOver = False
        
    def __init__(self):
        """Constructor"""
        self.action = numpy.zeros(5, int)
        self.isEpisodeOver = 0
        self.obs = None

    def getAction(self):
        if (self.isEpisodeOver):
            return numpy.ones(5, int)
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_d:
                    self.action[1] = 1
                elif event.key == pygame.K_a:
                    self.action[0] = 1
                elif event.key == pygame.K_s:
                    self.action[2] = 1
                if event.key == pygame.K_j:
                    self.action[self.KEY_JUMP] = 1
                if event.key == pygame.K_k:
                    self.action[self.KEY_SPEED] = 1
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_d:
                    self.action[1] = 0
                elif event.key == pygame.K_a:
                    self.action[0] = 0
                elif event.key == pygame.K_s:
                    self.action[2] = 0
                if event.key == pygame.K_j:
                    self.action[self.KEY_JUMP] = 0
                if event.key == pygame.K_k:
                    self.action[self.KEY_SPEED] = 0
        # print(self.action)
        return self.action


    def integrateObservation(self, obs):
        """This method stores the observation inside the agent"""
        # pdb.set_trace()
        if (len(obs) != 6):
            self.isEpisodeOver = True
        else:
            self.obs = obs

    def printObs(self):
        """for debug"""
        print repr(self.observation)
