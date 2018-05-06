import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def run_model(env, model, weight_file, render = False):
    avg_reward = []
    std_reward = []
    model.load_weights(weight_file)
    print("Load " + weight_file)
    for i in range(1):
        mu, std = evaluate(env, model, render=render)
        avg_reward.append(mu)
        std_reward.append(std)
        print("Finish %d/2 with mu: %f, std: %f" % (i, mu, std))
    return avg_reward, std_reward


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
    file_name = sys.argv[1]
    if file_name.startswith("eval"):
        eval_data = np.load(file_name)
        std_data = eval_data[:,1]
        mean_data = eval_data[:,0]
        win_data = eval_data[:,2]
        eval_len = eval_data.shape[0]

        eval_x = np.arange(0, eval_len*500, 500)

        f = plt.figure(1)
        errorfill(eval_x, mean_data, std_data)
        plt.xlabel('number of episodes')
        plt.ylabel('score of episode/evaluation')
        # plt.legend(handles=[eval_plot[0]], loc='upper left')

        g = plt.figure(2)
        plt.plot(eval_x, win_data, color='r')
        plt.xlabel('number of episodes')
        plt.ylabel('Winning rate over 100 evaluation episodes')
        plt.show()
    else:
        train_data = np.load(file_name)
        mean_data = train_data[:,0]
        eval_x = np.arange(0, len(mean_data))
        plt.plot(eval_x, mean_data, color='g')
        plt.xlabel('number of episodes')
        plt.ylabel('Training rewards')
        plt.show()
    

def render():
    env = gym.make('LunarLander-v2')
    # env = gym.make('CartPole-v0')
    # Load the policy model from file.
    model_file = sys.argv[1]
    with open(model_file, 'r') as f:
        model = keras.models.model_from_json(f.read())

    weight_file =  sys.argv[2]
    run_model(env, model, weight_file, render=True)


if __name__=="__main__":
    plot()




