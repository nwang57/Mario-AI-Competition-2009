__author__ = "Sergey Karakovskiy, sergey at idsia fullstop ch"
__date__ = "$May 12, 2009 11:18:19 PM$"

from experiment import Experiment


#class EpisodicExperiment(Experiment):
#    """ The extension of Experiment to handle episodic tasks. """
#
#    def doEpisodes(self, number = 1):
#        """ returns the rewards of each step as a list """
#        all_rewards = []
#        for dummy in range(number):
#            rewards = []
#            self.stepid = 0
#            # the agent is informed of the start of the episode
#            self.agent.newEpisode()
#            self.task.reset()
#            while not self.task.isFinished():
#                r = self._oneInteraction()
#                rewards.append(r)
#            all_rewards.append(rewards)
#        return all_rewards


from experiment import Experiment

class EpisodicExperiment(Experiment):
    """ The extension of Experiment to handle episodic tasks. 
        implement evaluation
    """

    def __init__(self, task, agent):
        super(EpisodicExperiment, self).__init__(task, agent)
        self.train_rewards = []
        self.cur_state = None
        self.all_states = []
        self.all_actions = []

    def reset(self):
        self.cur_state = None
        self.action = None
        self.stepid = 0
        self.agent.newEpisode()
        self.task.reset()
        self.done = False
    
    def doEpisodes(self, number = 1):
        """ returns the rewards of each step as a list """
        all_rewards = []
        all_obs = []
        all_action = []
        for dummy in range(number):
            rewards = []
            obvs = []
            actions = []
            self.stepid = 0
            # the agent is informed of the start of the episode
            self.agent.newEpisode()
            self.task.reset()
            while not self.task.isFinished():
                r, obs, action = self._oneInteraction()
                rewards.append(r)
                obvs.append(obs)
                actions.append(action)
                print (action)
            all_rewards.append(rewards)
            all_obs.extend(obvs)
            all_action.extend(actions)
            print('hey action')
            print (all_action)
        return all_rewards, all_obs, all_action


    def train(self, num_episodes = 100):
        for i in xrange(num_episodes):
            states = []
            actions = []
            self.reset()
            while True:
                self.stepid += 1
                raw_obs = self.task.getObservation()
                self.done = self.task.isFinished():
                if len(raw_obs) == 2:
                    next_state, reward = raw_obs
                else:
                    next_state = raw_obs
                # perform update with cur, action, next, reward
                if self.cur_state is not None:
                    states.append(self.cur_state)
                    actions.append(self.action)
                if self.done:
                    break
                self.cur_state = next_state
                self.agent.integrateObservation(self.cur_state)
                self.action = self.agent.getAction()
                self.task.performAction(self.action)
            all_obs.extend(states)
            all_action.extend(actions)
        

#class EpisodicExperiment(Experiment):
#    """
#    Documentation
#    """
#
#    statusStr = ("Loss...", "Win!")
#    agent = None
#    task = None
#
#    def __init__(self, agent, task):
#        """Documentation"""
#        self.agent = agent
#        self.task = task
#
#    def doEpisodes(self, amount):
#        for i in range(amount):
#            self.agent.newEpisode()
#            self.task.startNew()
#            while not self.task.isFinished():
#                obs = self.task.getObservation()
#                if len(obs) == 3:
#                    self.agent.integrateObservation(obs)
#                    self.task.performAction(self.agent.produceAction())
#                
#            r = self.task.getReward()
#            s = self.task.getStatus()
#            print "Episode #%d finished with status %s, fitness %f..." % (i, self.statusStr[s], r)
#            self.agent.giveReward(r)
