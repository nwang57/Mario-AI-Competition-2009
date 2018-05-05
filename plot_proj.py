import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def errorfill(x, y, yerr, color=None, alpha_fill=0.4, ax=None):
    ax = ax if ax is not None else plt.gca()
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr
    ax.plot(x, y, color=color)
    ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill)


def plot():
    op = sys.argv[1]
    print(op)
    if op == 'eval':
        eval_file = sys.argv[2]
        eval_data = np.load(eval_file)
        mean = eval_data[:,0]
        std = eval_data[:,1]
        percent = eval_data[:,2]
        eval_len = eval_data.shape[0]
        eval_x = np.arange(0, eval_len*100, 100)

        plt.figure()
        fig, ax1 = plt.subplots()
        ax1.errorbar(eval_x, mean, xerr=0, yerr=std, elinewidth=2, alpha=0.5)
        ax1.plot(eval_x, mean, 'b', linewidth=4)
        ax1.set_xlabel('number of episodes')
        ax1.set_ylabel('average reward per episodes')
        plt.show()

    if op == 'win':
        eval_file = sys.argv[2]
        eval_data = np.load(eval_file)
        mean = eval_data[:,0]
        std = eval_data[:,1]
        percent = eval_data[:,2]
        eval_len = eval_data.shape[0]
        eval_x = np.arange(0, eval_len*100, 100)

        plt.figure()
        fig, ax2 = plt.subplots()
        ax2.plot(eval_x, percent, 'r', linewidth=4)
        ax2.set_xlabel('number of episodes')
        ax2.set_ylabel('percent of winning per evaluation')
        plt.show()

    if op == 'training':
        training_file = sys.argv[2]
        training_data = np.load(training_file)
        #training_data = training_data[::50,:]
        #reward = training_data[:,0]
        reward = np.mean(np.reshape(training_data[:20000, 0], (50, 400)), axis=0)
        lengths = training_data[:,1]
        training_len = training_data.shape[0]
        #training_x = np.arange(0, training_len, 1)
        training_x = np.arange(0, len(reward), 1)

        plt.figure()
        fig, ax3 = plt.subplots()
        ax3.plot(training_x, reward, 'b', linewidth=2)
        ax3.set_xlabel('number of episodes')
        ax3.set_ylabel('total reward per episode')

        #ax4 = ax3.twinx()
        #ax4.plot(training_x, lengths, 'r', linewidth=2)
        #ax4.set_ylabel('steps taken before gameover/timeout')
        plt.show()

if __name__=="__main__":
    plot()




