__author__ = "Sergey Karakovskiy, sergey at idsia dot ch"
__date__ = "$Apr 30, 2009 1:46:32 AM$"

import sys

from experiments.episodicexperiment import EpisodicExperiment
from tasks.mariotask import MarioTask
from agents.forwardagent import ForwardAgent
from agents.humanAgent import HumanAgent
from agents.learningagent import LearningAgent
from agents.forwardrandomagent import ForwardRandomAgent
import pygame


#from pybrain.... episodic import EpisodicExperiment
#TODO: reset sends: vis, diff=, lt=, ll=, rs=, mariomode, time limit, pw,
# with creatures, without creatures HIGH.
# send creatures.

def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
    parser.add_argument('--agent',dest='train',type=str,default='human')
    parser.add_argument('--model',dest='model',type=str)
    parser.add_argument('--output', dest='output_file',type=str)
    parser.add_argument('--memory', dest='memory_mode',type=int)
    return parser.parse_args()

def main():
    args = parse_arguments()
    agent_name = args.agent
    render = args.render
    model = args.model

    task = MarioTask(initMarioMode = 2)

    agent = None
    if agent is 'human':
        agent = HumanAgent(task.ACTION_MAPPING, task.obs_space)
    elif agent is 'learning':
    	dim_obs = 39
    	dim_action = len(task.ACTION_MAPPING)
        agent = LearningAgent(dim_obs, dim_action, model)

    exp = EpisodicExperiment(task, agent)
    print 'Task Ready'
    exp.train(5)
    
    print "finished"

#    clo = CmdLineOptions(sys.argv)
#    task = MarioTask(MarioEnvironment(clo.getHost(), clo.getPort(), clo.getAgent().name))
#    exp = EpisodicExperiment(clo.getAgent(), task)
#    exp.doEpisodes(3)

if __name__ == "__main__":
    main()
else:
    print "This is module to be run rather than imported."
