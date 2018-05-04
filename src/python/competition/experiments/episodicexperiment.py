__author__ = "Sergey Karakovskiy, sergey at idsia fullstop ch"
__date__ = "$May 12, 2009 11:18:19 PM$"

from experiment import Experiment
import numpy as np
import keras


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

WIN_REWARD = 1000

class EpisodicExperiment(Experiment):
    """ The extension of Experiment to handle episodic tasks. 
        implement evaluation
    """

    def __init__(self, task, agent):
        super(EpisodicExperiment, self).__init__(task, agent)
        self.train_rewards = []
        self.cur_state = None
        self.action = None
        self.train_stat = []
        self.eval_stat = []
        self.config = agent.config

    def reset(self):
        self.cur_state = None
        self.action = None
        self.stepid = 0
        self.agent.reset()
        self.task.reset()
    
    def doEpisodes(self, number = 1):
        """ returns the rewards of each step as a list """
        all_rewards = []
        for dummy in range(number):
            rewards = []
            self.stepid = 0
            # the agent is informed of the start of the episode
            self.agent.newEpisode()
            self.task.reset()
            while not self.task.isFinished():
                r = self._oneInteraction()
                rewards.append(r)
            all_rewards.append(rewards)
        return all_rewards


    def train(self, num_episodes = 100, save_ep = 200, eval_ep=100):
        pretrain = self.config.DEMO_MODE
        for i in xrange(num_episodes):
            self.reset()
            reward_list = []
            if self.config.DEMO_MODE:
                if i < self.config.PRETRAIN_STEPS:
                    pretrain = True
                    self.agent.eps = 0.05
                elif i == self.config.PRETRAIN_STEPS:
                    pretrain = False
                    self.agent.eps = self.config.INITIAL_EPS
                    print("Pretrain finish")
            while True:
                self.stepid += 1
                raw_obs= self.task.getObservation()
                if len(raw_obs) == 2:
                    next_state, reward = raw_obs
                    if self.cur_state is not None:
                        self.agent.update_network(self.cur_state, self.action, reward, next_state, pretrain=pretrain)
                    if reward is None:
                        # epsisode finish
                        print("#{} Episode len {}, total rewards: {}, avg_reward: {}, eps: {}".format(i ,self.stepid, np.sum(reward_list), np.mean(reward_list), self.agent.eps))
                        self.train_stat.append([np.sum(reward_list), len(reward_list)])
                        reward_list = []
                        break
                    else:
                        reward_list.append(reward)
                else:
                    raise KeyError("obs len wrong")
                # perform update with cur, action, next, reward
                # if self.cur_state is not None:
                #     pass
                self.cur_state = next_state
                self.agent.integrateObservation(self.cur_state)
                self.action = self.agent.getAction()
                self.task.performAction(self.action)
            if (i % eval_ep == 0 or i == num_episodes-1):
                avg_r, std_r, win_percentage = self.eval()
                self.eval_stat.append([avg_r, std_r, win_percentage])
            if (i % save_ep == 0 or i == num_episodes-1):
                self.agent.save_weights(i)
                np.save("training_%d" % i, self.train_stat)
                np.save("eval_%d" % i, self.eval_stat)

    def eval(self, num_episodes=100):
        is_win = []
        total_rewards = []
        old_eps = self.agent.eps
        self.agent.eps = 0.05
        for i in xrange(num_episodes):
            self.reset()
            rewards = []
            while True:
                self.stepid += 1
                raw_obs= self.task.getObservation()
                if len(raw_obs) == 2:
                    next_state, reward = raw_obs
                    if reward is None:
                        total_rewards.append(np.sum(rewards))
                        win = (np.max(rewards) >= WIN_REWARD)
                        is_win.append((int)(win))
                        break
                    else:
                        rewards.append(reward)
                else:
                    raise KeyError("obs len wrong")
                self.cur_state = next_state
                self.agent.integrateObservation(self.cur_state)
                self.action = self.agent.getAction()
                self.task.performAction(self.action)
        self.agent.eps = old_eps
        return np.mean(total_rewards), np.std(total_rewards), np.mean(is_win)

######################################################## START IN CONSTRUCTION #####
    def to_onehot(self, val, dim):
        return keras.utils.to_categorical(val, num_classes=dim)
 
    def generate_episode_PG(self, dim_obs, dim_action):
        # Generates an episode by executing the current policy in the given env.
        # Returns:
        # - a list of states, indexed by time step
        # - a list of actions, indexed by time step
        # - a list of rewards, indexed by time step
        states = []
        actions = []
        rewards = []

        self.reset()
        raw_obs= self.task.getObservation()
        assert(len(raw_obs) == 2)
        state = np.reshape(raw_obs[0],(1,dim_obs))

        while True:
            states.append(state)
            self.agent.integrateObservation(state)
            action = self.agent.getAction()
            actions.append(action)
            self.task.performAction(action)
            raw_obs= self.task.getObservation()
            if len(raw_obs) == 2:
                next_state, reward = raw_obs
                if reward is None:
                    break
            rewards.append(reward)
            state = np.reshape(next_state, (1,dim_obs))

        actions = self.to_onehot(actions, dim_action)
        assert(len(states) == len(actions))
        assert(len(states) == len(rewards) + 1)
        return np.concatenate(states[:-1]), actions[:-1], np.array(rewards)

    def train_PG(self, dim_obs, dim_action, num_episodes=100, gamma=0.99, save_ep = 1000, eval_ep=500):
        # Trains the model on a single episode using A2C.
        n_ep = 0
        while n_ep < num_episodes:
            states, actions, rewards = self.generate_episode_PG(dim_obs, dim_action)
            print("#{} Episode len {}, total rewards: {}, avg_reward: {}".format(n_ep ,len(rewards), np.sum(rewards), np.mean(rewards)))
            self.train_stat.append([np.sum(rewards), len(rewards)])
            self.agent.update_network(n_ep, states, actions, rewards, gamma)
            if (n_ep % eval_ep == 0 or n_ep == num_episodes-1):
                avg_r, std_r, win_percentage = self.eval_PG(dim_obs, dim_action)
                self.eval_stat.append([avg_r, std_r, win_percentage])
            if (n_ep % save_ep == 0 or n_ep == num_episodes-1):
                self.agent.save_model_weights(n_ep)
                np.save("training_%d" % n_ep, self.train_stat)
                np.save("eval_%d" % n_ep, self.eval_stat)
            n_ep += 1

    def eval_PG(self,dim_obs, dim_action, num_epi = 100):
        rewards = []
        is_win = []
        for e in range(num_epi):
            _, _, r = self.generate_episode_PG(dim_obs, dim_action)
            rewards.append(np.sum(r))
            win = (np.max(r) >= WIN_REWARD)
            is_win.append((int)(win))
        return np.mean(rewards), np.std(rewards), np.mean(is_win)
########################################################## END IN CONSTRUCTION #####

    def run(self, num_episodes = 100):
        for i in xrange(num_episodes):
            self.reset()
            while True:
                self.stepid += 1
                raw_obs = self.task.getObservation()
                if len(raw_obs) == 2:
                    next_state, reward = raw_obs
                    if self.cur_state is not None:
                        self.agent.record(self.cur_state, self.action, reward, next_state)
                    if reward is None:
                        # episode finish
                        break
                else:
                    raise KeyError("obs len wrong")
                self.cur_state = next_state
                self.action = self.agent.getAction()
                self.task.performAction(self.action)             
        

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
