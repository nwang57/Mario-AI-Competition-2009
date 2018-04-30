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

    def __init__(self, action_mapping, num_output):
        """Constructor"""
        pygame.init()
        pygame.display.set_mode([1,1])
        self.action = numpy.zeros(6, int)
        self.isEpisodeOver = 0
        self.obs = None
        self.reverse_map = {}
        for k,v in action_mapping.items():
            self.reverse_map[tuple(v)] = k

    def reset(self):
        self.action = numpy.zeros(6, int)
        self.isEpisodeOver = 0
        self.obs = None

    def getAction(self):
        if (self.isEpisodeOver):
            return numpy.ones(6, int)
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
                if event.key == pygame.K_b:
                    import pdb;pdb.set_trace()
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
        key = tuple(self.action)
        if key not in self.reverse_map:
            return 0
        else:
            return self.reverse_map[key]


    def integrateObservation(self, obs):
        """This method stores the observation inside the agent"""
        # pdb.set_trace()
        if (len(obs) == 5):
            # print(obs)
            self.isEpisodeOver = True
        else:
            self.obs = obs[0]
            self.reward = obs[1]

    def printObs(self):
        """for debug"""
        print repr(self.observation)
