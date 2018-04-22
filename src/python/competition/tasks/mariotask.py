__author__ = "Sergey Karakovskiy, sergey at idsia fullstop ch"
__date__ = "$May 7, 2009 12:47:18 PM$"

from client.marioenvironment import MarioEnvironment
from episodictask import EpisodicTask

if __name__ != "__main__":
    print "Loading %s ..." % __name__;

class MarioTask(EpisodicTask):
    """Encapsulates Mario specific options and transfers them to EpisodicTask"""
    ACTION_MAPPING = {
        0: [0,0,0,0,0], #stay
        1: [0,0,0,1,0], #jump
        2: [0,0,0,0,1], #fire
        3: [0,0,0,1,1], #jump + fire
        4: [1,0,0,0,0], #left
        5: [1,0,0,1,0], #left jump
        6: [1,0,0,0,1], #left + speed
        7: [1,0,0,1,1], # left + jump + speed
        8:  [0,1,0,0,0], #right
        9: [0,1,0,1,0], #right jump
        10: [0,1,0,0,1], # right speed
        11: [0,1,0,1,1], # right jump speed
    }

    def __init__(self, *args, **kwargs):
        EpisodicTask.__init__(self, MarioEnvironment(*args, **kwargs))
        self.action_space = len(self.ACTION_MAPPING)
        #self.reset()
    
    def reset(self):
        EpisodicTask.reset(self)
    #        sleep(3)
    #        EpisodicTask.reset(self)
    #        sleep(3)
        self.finished = False
        self.epi_reward = 0
        self.cur_reward= 0
        self.status = 0       

    def isFinished(self):
        return self.finished

    def getObservation(self):
        # if len(obs) = 2 : (state, reward)
        obs = EpisodicTask.getObservation(self)
        if len(obs) == MarioEnvironment.numberOfFitnessValues:
            self.epi_reward = obs[1]
            self.status = obs[0]
            self.finished = True
        elif len(obs) == 2:
            # obs = (state, reward)
            self.cur_reward = obs[1]
            return obs
        return obs
    
    def performAction(self, action):
        if not self.isFinished():
            if type(action) is int:
                EpisodicTask.performAction(self, self.ACTION_MAPPING[action])           
            else:
                EpisodicTask.performAction(self, action)           

    def getReward(self):
        """ custome reward for each transition """
        return self.cur_reward

    def getEpisodeReward(self):
        """ Fitness gained on the level """
        return self.epi_reward
    
    def getWinStatus(self):
        return self.status
