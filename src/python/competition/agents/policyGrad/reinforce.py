import sys
import argparse
import numpy as np
import tensorflow as tf
import keras


import functools

def lazy_property(func):
    attribute = '_lazy_' + func.__name__

    @property
    @functools.wraps(func)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, func(self))
        return getattr(self, attribute)
    return wrapper

class Reinforce(object):
    # Implementation of the policy gradient method REINFORCE.

    def __init__(self, model, lr, output_file="model", weight_file=None):
        self.model = model

        # weight file
        self.output_file = output_file
        self.weight_file = weight_file
        if self.weight_file is not None:
            self.model.load_weights(self.weight_file)

        self.lr = lr
        # TODO: Define any training operations and optimizers here, initialize
        #       your variables, or alternately compile your model here.

        self.sess = keras.backend.get_session()
        self.graph = self.sess.graph
        self.output = self.model.output
        self.input = self.model.inputs[0]

        self.input_dim = self.input.shape[1]
        self.output_dim = self.output.shape[1]
        self.G = tf.placeholder(tf.float32, shape=[None, 1])
        self.A = tf.placeholder(tf.float32, shape=[None, self.output_dim])

        self.gamma = 0.99

        # checkpoint
        self.is_checkpoint = True
        self.checkpoint_freq = 5000

        self.eval_freq = 200
        self.eval_avg_reward = []
        self.train_reward = []

        self.loss
        self.optimize

    def save_weights(self, n_ep):
        self.model.save_weights("%s_%d.h5" % (self.output_file, n_ep))

    def load_weights(self):
        self.model.load_weights(self.weight_file)

    @lazy_property
    def optimize(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        return optimizer.minimize(self.loss), optimizer.compute_gradients(self.loss)

    @lazy_property
    def loss(self):
        # return -tf.reduce_sum(tf.multiply(self.G, tf.log(tf.reduce_sum(tf.multiply(self.A, self.output), axis=1, keepdims=True))))
        return -tf.matmul(tf.transpose(self.G), tf.log(tf.reduce_sum(tf.multiply(self.A, self.output), axis=1, keepdims=True)))[0,0]

    def get_G(self, rewards):
        rewards = rewards
        T = len(rewards)
        G = np.zeros((T,1))
        G[T-1,0] = rewards[T-1]
        for i in range(T-2,-1,-1):
            G[i,0] = G[i+1,0]*self.gamma + rewards[i]
        return (G - np.mean(G))/np.std(G)

    def train(self, env, num_episodes):
        # Trains the model on a single episode using REINFORCE.
        # TODO: Implement this method. It may be helpful to call the class
        #       method generate_episode() to generate training data.
        n_ep = 1
        batch_episode = 1
        input_states = []
        input_actions = []
        input_G = []
        _, _, X = self.evaluate(env, 100)
        while n_ep < num_episodes:
            states, actions, rewards = self.generate_episode(env)
            G = self.get_G(rewards)
            #batch_size = 32
            #num_epoches = 5 * (T // batch_size)
            #for i in range(num_epoches):
            #    index = np.random.choice(T, batch_size)
            #    g = G[:, index]
            #    a = actions[index, :]
            #    s = states[index, :]
            #    _, grads, loss = self.sess.run([self.optimize[0], self.optimize[1], self.loss], feed_dict = {self.A: a, self.input: s, self.G: g})
            if (n_ep < 10000):
                G -= X
            if n_ep % batch_episode == 0:
                input_states.append(states)
                input_actions.append(actions)
                input_G.append(G)
                _, grads, last_relu, o, loss = self.sess.run([self.optimize[0], self.optimize[1], self.model.layers[-2].output, self.output, self.loss], feed_dict = {self.A: np.concatenate(input_actions), self.input: np.concatenate(input_states), self.G: np.concatenate(input_G)})
                input_states = []
                input_actions = []
                input_G = []
            else:
                input_states.append(states)
                input_actions.append(actions)
                input_G.append(G)
            if n_ep % self.eval_freq == 0 or n_ep == num_episodes-1:
                avg_reward, std_reward, X = self.evaluate(env, 50)
                self.eval_avg_reward.append(avg_reward)
                print('Iter %d, objective is %f, average reward is %f, std is %f' % (n_ep, -loss, avg_reward, std_reward))
            if self.is_checkpoint and (n_ep % self.checkpoint_freq == 0 or n_ep == num_episodes-1):
                self.save_weights(n_ep)
                print(o)
            self.train_reward.append(np.sum(rewards))
            n_ep += 1
        # save training and eval data points
        np.save("taining", self.train_reward)
        np.save("eval", self.eval_avg_reward)

    def evaluate(self, env, num_eps=100):
        average_reward = []
        X = []
        for j in range(num_eps):
            _, _, r = self.generate_episode(env)
            X.append(np.mean(self.get_G(r)))
            average_reward.append(np.sum(r))
        return np.mean(average_reward), np.std(average_reward), np.mean(X)

    @staticmethod
    def get_action(model, state):
        probs = model.predict(state).flatten()
        return np.random.choice(probs.size,p=probs)

    @staticmethod
    def to_onehot(val, dim):
        # :param val: a list of indices
        # :param dim: number of cols
        # :ret: a matrix with len(val) * dim one hot vector
        return keras.utils.to_categorical(val, num_classes=dim)

    def generate_episode(self, env, render=False):
        # Generates an episode by executing the current policy in the given env.
        # Returns:
        # - a list of states, indexed by time step
        # - a list of actions, indexed by time step
        # - a list of rewards, indexed by time step
        # TODO: Implement this method.
        states = []
        actions = []
        rewards = []
        obs_dim = env.observation_space.shape[0]

        done = False
        state = np.reshape(env.reset(),(1,obs_dim))

        while not done:
            if render:
                env.render()
            states.append(state)
            action = Reinforce.get_action(self.model, state)
            actions.append(action)
            next_state, reward, done, _ = env.step(action)
            rewards.append(reward)
            state = np.reshape(next_state, (1,obs_dim))

        actions = Reinforce.to_onehot(actions, env.action_space.n)
        return np.concatenate(states), actions, np.array(rewards)


def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-config-path', dest='model_config_path',
                        type=str, default='LunarLander-v2-config.json',
                        help="Path to the model config file.")
    parser.add_argument('--num-episodes', dest='num_episodes', type=int,
                        default=50000, help="Number of episodes to train on.")
    parser.add_argument('--lr', dest='lr', type=float,
                        default=5e-4, help="The learning rate.")
    parser.add_argument('--output', dest='output_file',
                        type=str)
    parser.add_argument('--weight_file', dest='weight_file',
                        type=str)

    # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    parser_group = parser.add_mutually_exclusive_group(required=False)
    parser_group.add_argument('--render', dest='render',
                              action='store_true',
                              help="Whether to render the environment.")
    parser_group.add_argument('--no-render', dest='render',
                              action='store_false',
                              help="Whether to render the environment.")
    parser.set_defaults(render=False)

    return parser.parse_args()


def main(args):
    np.random.seed(661)

    # Parse command-line arguments.
    args = parse_arguments()
    model_config_path = args.model_config_path
    num_episodes = args.num_episodes
    lr = args.lr
    render = args.render

    # Create the environment.
    env = gym.make('LunarLander-v2')
    # env = gym.make('CartPole-v0')
    # Load the policy model from file.
    with open(model_config_path, 'r') as f:
        model = keras.models.model_from_json(f.read())
    print(args.output_file, args.weight_file)
    # TODO: Train the model using REINFORCE and plot the learning curve.
    my_reinforce = Reinforce(model, lr, output_file=args.output_file, weight_file=args.weight_file)
    my_reinforce.train(env, num_episodes)

if __name__ == '__main__':
    main(sys.argv)
