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
import argparse


#from pybrain.... episodic import EpisodicExperiment
#TODO: reset sends: vis, diff=, lt=, ll=, rs=, mariomode, time limit, pw,
# with creatures, without creatures HIGH.
# send creatures.

def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
    parser.add_argument('--agent',dest='agent',type=str,default='human')
    parser.add_argument('--model',dest='model',type=str)
    parser.add_argument('--n',dest='num_epi',type=int,default='100')
    parser.add_argument('--output', dest='output_file',type=str)
    parser.add_argument('--memory', dest='memory_mode',type=int)
    return parser.parse_args()

def main():
    args = parse_arguments()
    agent_name = args.agent
    model = args.model

    task = MarioTask(initMarioMode = 2)

    print agent_name
    if agent_name == 'human':
        agent = HumanAgent(task.ACTION_MAPPING, task.obs_space)
        exp = EpisodicExperiment(task, agent)
        exp.run(5)
    elif agent_name == 'learning':
        print "hello"
    	dim_obs = 39
    	dim_action = len(task.ACTION_MAPPING)
        agent = LearningAgent(dim_obs, dim_action, model)
        exp = EpisodicExperiment(task, agent)
        exp.train(args.num_epi)
    elif agent_name == 'pg':
    	dim_obs = 39
    	dim_action = len(task.ACTION_MAPPING)
        model_config_path = 'agents/policyGrad/mario_config.json'
        with open(model_config_path, 'r') as f:
            model_config = keras.models.model_from_json(f.read())
        lr = 0.0005
        critic_lr = 0.0005
        n = 50
        agent = PGAgent(dim_obs, dim_action, model_config, lr, critic_lr, n)
        exp = EpisodicExperiment(task, agent)
        exp.train_PG(dim_obs, dim_action)
    
    print "finished"

#    clo = CmdLineOptions(sys.argv)
#    task = MarioTask(MarioEnvironment(clo.getHost(), clo.getPort(), clo.getAgent().name))
#    exp = EpisodicExperiment(clo.getAgent(), task)
#    exp.doEpisodes(3)

if __name__ == "__main__":
    main()
else:
    print "This is module to be run rather than imported."
