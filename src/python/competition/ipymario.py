__author__ = "Sergey Karakovskiy, sergey at idsia dot ch"
__date__ = "$Apr 30, 2009 1:46:32 AM$"

import sys

from experiments.episodicexperiment import EpisodicExperiment
from tasks.mariotask import MarioTask
from agents.forwardagent import ForwardAgent
from agents.humanAgent import HumanAgent
from agents.learningagent import LearningAgent
from agents.forwardrandomagent import ForwardRandomAgent
from agents.policyGrad.pgAgent import PGAgent
import pygame
import argparse


#from pybrain.... episodic import EpisodicExperiment
#TODO: reset sends: vis, diff=, lt=, ll=, rs=, mariomode, time limit, pw,
# with creatures, without creatures HIGH.
# send creatures.

def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
    parser.add_argument('--agent',dest='agent',type=str,default='human')
    parser.add_argument('--pg_model',dest='pg_model',type=str)
    parser.add_argument('--model',dest='model',type=str)
    parser.add_argument('--n',dest='num_epi',type=int,default='100')
    parser.add_argument('--memory', dest='memory_mode',type=int)
    parser.add_argument('--prefix', dest='prefix', type=str)
    parser.add_argument('--save_epi', dest='save_epi', type = int, default=1000)
    parser.add_argument('--actor_weights', dest='actor_weights', type=str)
    parser.add_argument('--critic_weights', dest='critic_weights', type=str)
    return parser.parse_args()

def main():
    args = parse_arguments()
    agent_name = args.agent
    model = args.model
    fn_prefix = args.prefix
    save_epi = args.save_epi
    actor_weights = args.actor_weights
    critic_weights = args.critic_weights

    task = MarioTask(initMarioMode = 0)
    dim_obs = 54
    dim_action = len(task.ACTION_MAPPING)

    print agent_name
    if agent_name == 'human':
        agent = HumanAgent(task.ACTION_MAPPING, task.obs_space)
        exp = EpisodicExperiment(task, agent)
        exp.run(200)
    elif agent_name == 'learning':
        agent = LearningAgent(dim_obs, dim_action, model)
        exp = EpisodicExperiment(task, agent)
        exp.train(args.num_epi, save_ep=save_epi)
    elif agent_name == 'pg':
        model_config_path = args.pg_model
        lr = 0.001
        critic_lr = 0.001
        n = 100
        gamma = 0.9
        agent = PGAgent(dim_obs, dim_action, model_config_path, lr, critic_lr, n, output_file = fn_prefix, actor_file=actor_weights, critic_file=critic_weights)
        exp = EpisodicExperiment(task, agent)
        exp.train_PG(dim_obs, dim_action, args.num_epi, gamma, save_ep=save_epi)
    
    print "finished"

#    clo = CmdLineOptions(sys.argv)
#    task = MarioTask(MarioEnvironment(clo.getHost(), clo.getPort(), clo.getAgent().name))
#    exp = EpisodicExperiment(clo.getAgent(), task)
#    exp.doEpisodes(3)

if __name__ == "__main__":
    main()
else:
    print "This is module to be run rather than imported."
