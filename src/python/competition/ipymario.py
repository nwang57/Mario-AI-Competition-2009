__author__ = "Sergey Karakovskiy, sergey at idsia dot ch"
__date__ = "$Apr 30, 2009 1:46:32 AM$"

import sys

from experiments.episodicexperiment import EpisodicExperiment
from tasks.mariotask import MarioTask
from agents.forwardagent import ForwardAgent
from agents.humanAgent import HumanAgent
from agents.forwardrandomagent import ForwardRandomAgent
import pygame
import pickle
import argparse



#from pybrain.... episodic import EpisodicExperiment
#TODO: reset sends: vis, diff=, lt=, ll=, rs=, mariomode, time limit, pw,
# with creatures, without creatures HIGH.
# send creatures.

def main():
    args = parse_arguments()
    agent_name = args.agent_name
    filename = args.filename
    num_epi = args.num_epi
    print(agent_name, filename, num_epi)
    pygame.init()
    pygame.display.set_mode([1,1])
    task = MarioTask(initMarioMode = 2)
    if agent_name == 'human' :
        agent = HumanAgent(task.ACTION_MAPPING)
    else:
        agent = Forwardagent()
    
    exp = EpisodicExperiment(task, agent)
    print 'Task Ready'
    exp.train(num_epi)
    print 'mm 2:', task.reward
    
    if agent_name == 'human':
        print (all_action)
        with open('./expert_data/' + filename + '_demo.pckl', 'wb') as f:
            pickle.dump((exp.all_states, exp.all_actions), f)
    print "finished"
    pygame.quit()
#    clo = CmdLineOptions(sys.argv)
#    task = MarioTask(MarioEnvironment(clo.getHost(), clo.getPort(), clo.getAgent().name))
#    exp = EpisodicExperiment(clo.getAgent(), task)
#    exp.doEpisodes(3)

def parse_arguments():
    parser = argparse.ArgumentParser("Super Mario Agent Argument Parser")
    parser.add_argument('--agent',dest='agent_name',type=str, default = 'human')
    parser.add_argument('--filename',dest='filename',type=str)
    parser.add_argument('--epi', dest='num_epi', type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    main()
else:
    print "This is module to be run rather than imported."
