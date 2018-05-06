#!/usr/bin/env python
from __future__ import print_function
import tensorflow as tf
import numpy as np
import h5py, sys, copy, argparse
from numpy import random
import math
import json
from memory import prioritizedMemory, classicMemory
from config import Config
import functools
import time


'''
1. segtree needs to find values from idxes
2. calculate and store n_step reward and q_function
3. copy weight not sure if correct
4. separate demo and replay buffer weights
'''

def lazy_property(func):
    attribute = '_lazy_' + func.__name__

    @property
    @functools.wraps(func)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, func(self))
        return getattr(self, attribute)
    return wrapper


class Model():

    def __init__(self, num_input, num_output, config, model_name, output_file='model', weight_file=None):

        self.sess = tf.Session()
        self.num_input = num_input
        self.num_output = num_output
        self.config = config
        self.output_file = output_file
        self.model_name = model_name
        self.demo_mode = self.config.DEMO_MODE
        self.select_net_name = 'select_net'
        self.target_net_name = 'eval_net'
        self.input = tf.placeholder(tf.float32, name= 'input_ph', shape=[None, self.num_input])
        self.target_t1 = tf.placeholder(tf.float32, name= 'output_ph', shape=[None, self.num_output])
        self.action_t1 = tf.placeholder(tf.int32, name="action_t1", shape=[None])
        self.is_demo = tf.placeholder(tf.float32, name="is_demo", shape=[None])
        self.weight_file = weight_file

        if self.config.PRIORITIZED:
            self.memory = prioritizedMemory(config.MEMORY_SIZE, config.BURN_IN_SIZE)
        else: 
            self.memory = classicMemory(config.MEMORY_SIZE, config.BURN_IN_SIZE)

        # if self.demo_mode:
        #     self.target_tn = tf.placeholder(tf.float32, name= 'output_tn', shape=[None, self.num_output])
        if self.demo_mode:
            self.demo = self.get_demo_memory(self.config.DEMO_FILE)
            print("demo size: %d" % self.demo.cur_size)

        self.model
        self.loss
        self.optimize
        self.abs_err

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        if self.weight_file is not None:
            self.load_weights(self.weight_file)

    def get_demo_memory(self, file_name):
        start = time.clock()
        data = np.load(file_name)
        size = len(data)
        if self.config.PRIORITIZED:
            memory = prioritizedMemory(size)
        else:
            memory = classicMemory(size)
        for transition in data:
            memory.append(transition)
        print("Memory load spends %s" % (time.clock() - start))
        return memory

        

    @lazy_property
    def model(self):
        if self.model_name == 'dqn':
            return self.dqn
        if self.model_name == 'ddqn':
            return self.ddqn

    @lazy_property
    def dqn(self):
        return (self.get_layers("dqn"), None)

    def get_layers(self, scope_name):
        num_hidden = 32
        with tf.variable_scope(scope_name):
            layer1 = tf.layers.dense(name = 'l1', inputs = self.input, units = 80, activation=tf.nn.relu)
            layer2 = tf.layers.dense(name = 'l2', inputs = layer1, units = num_hidden, activation=tf.nn.relu)
            layer3 = tf.layers.dense(name = 'l3', inputs = layer2, units = num_hidden, activation=tf.nn.relu)
            outlayer = tf.layers.dense(name = 'l4', inputs = layer3, units = self.num_output)
        return outlayer

    @lazy_property
    def ddqn(self):
        l1_out = self.get_layers(self.select_net_name)
        l2_out = self.get_layers(self.target_net_name)
        return (l1_out, l2_out)

    @lazy_property
    def update_target_network(self):
        weight_select = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.select_net_name)
        weight_target = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.target_net_name)
        return [tf.assign(t, s) for t,s in zip(weight_target, weight_select)]


    @lazy_property   
    def loss(self): 
        if self.demo_mode:
            j_1 = tf.reduce_mean(self.one_step_loss, name='one_step_loss')
            # j_n = tf.reduce_mean(self.n_step_loss, name='n_step_loss')
            j_e = tf.reduce_mean(self.loss_e,  name='sup_loss')
            j_l2 = tf.reduce_sum([tf.reduce_mean(reg_l) for reg_l in tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)], name='reg_loss')
            return j_1 + j_e * self.config.LAMBDAS[1] + j_l2 * self.config.LAMBDAS[2]
        else: #ddqn or dqn
            return tf.reduce_mean(self.one_step_loss)

    @lazy_property
    def abs_err(self):
        if self.demo_mode:
            j_1 = tf.reduce_sum(self.one_step_loss, axis=1, name='one_step_loss_sum')
            # j_n = tf.reduce_sum(self.n_step_loss, axis=1,name='n_step_loss_sum')
            j_e = self.loss_e
            return j_1 + j_e * self.config.LAMBDAS[1]
        else:
            return tf.reduce_sum(self.one_step_loss, axis=1)

    @lazy_property
    def one_step_loss(self):
        # return an target_t1 shape loss (batch_size, num_actions)
        return tf.losses.mean_squared_error(self.target_t1, self.model[0], reduction=tf.losses.Reduction.NONE)

    # @lazy_property
    # def n_step_loss(self):
    #     # return an target_t1 shape loss (batch_size, num_actions)
    #     return tf.losses.mean_squared_error(self.target_tn, self.model[0], reduction=tf.losses.Reduction.NONE)

    def l_e(self, a, ae):
        ret = tf.cond(tf.equal(a,ae), lambda: 0.0, lambda: self.config.JE_IF_DIFFER, name='is_equal')
        return ret

    @lazy_property
    def action_list(self):
        return tf.constant(range(self.num_output))

    def add_margin(self, actions, i, a):
        return self.model[0][i][a] + self.l_e(a, actions[i])


    @lazy_property
    def loss_e(self):
        # return an target_t1 shape loss (batch_size)
        actions = self.action_t1

        bias_mask = tf.reduce_max(self.model[0] + tf.multiply((tf.ones([self.config.BATCH_SIZE, self.num_output])
            - tf.one_hot(actions, self.num_output)),
            tf.constant(self.config.JE_IF_DIFFER)), axis=1, keepdims=True)
        pad = tf.reshape(tf.range(self.config.BATCH_SIZE), [-1, 1])
        indices = tf.concat([pad, tf.reshape(actions, [-1,1])], axis=1)
        loss_je = bias_mask - tf.reshape(tf.gather_nd(self.model[0], indices), [-1, 1])
        # self.l_e(self.action_list, actions) + self.model[0]
        

        #for i in range(self.config.BATCH_SIZE):
        #    f = lambda a: self.add_margin(actions, i, a)
        #    max_value = tf.reduce_max(tf.map_fn(f, self.action_list,dtype=tf.float32), name='max')
        #    loss_je.append(self.is_demo[i] * (max_value - self.model[0][i][actions[i]]))
        #return tf.convert_to_tensor(loss_je, dtype=tf.float32)
        return loss_je

    @lazy_property
    def optimize(self):
        optimizer = tf.train.AdamOptimizer(learning_rate = self.config.LEARNING_RATE)
        return optimizer.minimize(self.loss), optimizer.compute_gradients(self.loss)


    def perceive(self,transition):
        return self.memory.append(transition)


    def train(self, n_iter, pretrain=False):
        # start = time.clock()
        if self.demo_mode:
            self.train_with_demo(pretrain=pretrain)
        else:
            self.train_no_demo()
        # print("one step spends %s" %(time.clock() - start))
        if self.model_name == 'ddqn':
            self.update_target(n_iter)

        # writer = tf.summary.FileWriter('./log', self.sess.graph)
        # writer.close()


    def train_no_demo(self):
        batch_ = self.sample_batch(self.config.BATCH_SIZE)
        s_t0, a_t0, r_t1, s_t1, d_t0, idxs, _ = batch_
        old_q_values = self.sess.run(self.model[0], feed_dict={self.input: s_t0})
        new_q_values = self.sess.run(self.model[0], feed_dict={self.input: s_t1})

        if self.model_name == 'ddqn':
            target = True
            Y_target = self.sess.run(self.model[1], feed_dict={self.input: s_t1})
        else:
            target = False

        for i in range(self.config.BATCH_SIZE):
            y_target = Y_target[i] if target else None
            old_q_values[i][a_t0[i]] = self.get_target(r_t1[i],new_q_values[i],d_t0[i], target=target, Y_target=y_target) 
        _, abs_err = self.sess.run([self.optimize[0], self.abs_err],feed_dict={self.target_t1: old_q_values, self.input: s_t0})
        
        if self.config.PRIORITIZED:
            for i in range(self.config.BATCH_SIZE):
                self.memory.change_weight(idxs[i], abs(abs_err[i]))
                

    def train_with_demo(self, pretrain=False):

        s_t0, a_t0, r_t1, s_t1, d_t0, idxs, is_demo = self.sample_batch(self.config.BATCH_SIZE, pretrain=pretrain)
        q_t0 = self.sess.run(self.model[0], feed_dict={self.input: s_t0})
        #for td-n update
        q_t0_u1 = np.copy(q_t0)
        # q_t0_un = np.copy(q_t0)

        q_t1 = self.sess.run(self.model[0], feed_dict={self.input: s_t1})
        # q_tn = self.sess.run(self.model[0], feed_dict={self.input: s_tn})

        if self.model_name == 'ddqn':
            target = True
            Y_target_t1 = self.sess.run(self.model[1], feed_dict={self.input: s_t1})
            # Y_target_tn = self.sess.run(self.model[1], feed_dict={self.input: s_tn})
        else:
            target = False

        for i in range(self.config.BATCH_SIZE):
            y_target_t1 = Y_target_t1[i] if target else None
            # y_target_tn = Y_target_tn[i] if target else None

            q_t0_u1[i][a_t0[i]] = self.get_target(r_t1[i], q_t1[i], False, target=target, Y_target=y_target_t1)
            # calculate n-step TD target
            # target = r_t+1 + gamma*r_t+2 + ... + gamma^n-1 * r_t+n + gamma^n * q_n
            # q_t0_un[i][a_t0[i]] = self.get_target(discounted_r[i], q_tn[i], d_t0[i], target=target, Y_target=y_target_tn, step=self.config.N_STEP)
        _, abs_err = self.sess.run([self.optimize[0], self.abs_err],feed_dict={
                                                       self.target_t1: q_t0_u1, 
                                                       self.input: s_t0,
                                                       self.action_t1: a_t0,
                                                       self.is_demo: is_demo})
        print("Abs error is %f" % abs_err)
        # _, abs_err = self.sess.run([self.optimize[0], self.abs_err],feed_dict={
        #                                                self.target_t1: q_t0_u1, 
        #                                                self.target_tn: q_t0_un, 
        #                                                self.input: s_t0,
        #                                                self.action_t1: a_t0,
        #                                                self.is_demo: is_demo})
        # print("One step")
        # new_q_t0 = self.sess.run(self.model[0], feed_dict={self.input: s_t0})
        # self.print_target(q_t0, q_t0_u1, q_t0_un,new_q_t0)
        # print("loss: %.3f, %.3f, %.3f, %.3f" % (np.mean(sup_loss), np.mean(one_step), np.mean(n_step), np.mean(loss)))
        if self.config.PRIORITIZED:
            for i in range(self.config.BATCH_SIZE):
                if is_demo[i] == 1:
                    self.demo.change_weight(idxs[i], abs(abs_err[i] + self.config.DEMO_BONUS))
                else:
                    self.memory.change_weight(idxs[i], abs(abs_err[i]))
            #todo: update weight



    def sample_batch(self, batch_size, pretrain=False): #improvement: append source to sample
        
        if self.demo_mode: 
            if pretrain:
                sample, idxs = self.demo.sample(batch_size)
                num_demo = batch_size
            else: 
                # demo_weight = self.demo.get_total_weight()
                # replay_weight = self.memory.get_total_weight()
                # prob_demo = demo_weight / (demo_weight + replay_weight)
                demo_num = int (batch_size * 0.5)
                replay_num = batch_size - demo_num
                # import pdb;pdb.set_trace()

                sample_d, idxs_d = self.demo.sample(demo_num)
                sample_r, idxs_r = self.memory.sample(replay_num)
                
                if demo_num == 0:
                    transition_dim = sample_r.shape[1]
                    sample_d = sample_d.reshape((0, transition_dim))
                    idxs_d = idxs_d.reshape((0, ))
                elif demo_num == batch_size:
                    transition_dim = sample_d.shape[1]
                    sample_r = sample_r.reshape((0, transition_dim))
                    idxs_r = idxs_r.reshape((0, ))

                sample = np.concatenate((sample_d, sample_r))
                idxs = np.concatenate((idxs_d, idxs_r)).astype('int64')
                num_demo = demo_num

        else:
            sample, idxs = self.memory.sample(batch_size)
            num_demo = 0
        # print("demo size: %d" % num_demo)
        s_t0 = np.stack(sample[:,0])
        a_t0 = np.stack(sample[:,1])
        r_t1 = np.stack(sample[:,2])
        s_t1 = np.stack(sample[:,3])
        d_t0 = np.stack(sample[:,4])
        is_demo = np.zeros(self.config.BATCH_SIZE)
        is_demo[range(num_demo)] = 1

        # if self.demo_mode:
        #     discounted_r = np.stack(sample[:,5])
        #     s_tn = np.stack(sample[:,6])
        #     return s_t0, a_t0, r_t1, s_t1, d_t0, discounted_r, s_tn, idxs, is_demo
        # else: 
        return s_t0, a_t0, r_t1, s_t1, d_t0, idxs, is_demo

    def get_target(self, reward, q_vals, done, target=False, Y_target=None, step=1):
        if done:
            y = reward
        else:
            if target:
                y = reward + (self.config.GAMMA**step) * Y_target[np.argmax(q_vals)]
            else:
                y = reward + (self.config.GAMMA**step) * np.max(q_vals)
        return y 


    def update_target(self, n_iter):
        if n_iter % self.config.UPDATE_TARGET_FREQ == 0:
            self.sess.run(self.update_target_network)


    def predict(self, state):
        return self.model[0].eval(session=self.sess, feed_dict = {self.input: state.reshape(1,self.num_input)})

    def printPolicyMountainCar(self):
        x, y = np.mgrid[0.06:-0.06:8j, -1.1:0.4:8j]
        states = np.vstack([x.ravel(), y.ravel()])
        y = self.model[0].eval(feed_dict={self.input: states.T}, session = self.sess)
        #print(states)
        #print(y)
        actions = np.argmax(y, axis=1)
        print_array = [' < ', ' - ', ' > ']
        for i in range(8):
            for j in range(8):
                print(print_array[actions[i * 3 + j]], end='')
                print(y[i * 3 + j,:].max(), end="")
            print()

    def print_target(self, q_t0, q_t0_u1, q_t0_un,new_q_t0):
        n, m = q_t0.shape
        for i in range(n):
            for j in range(m):
                print("%.3f:%.3f:%.3f:%.3f\t" % (q_t0[i][j], q_t0_u1[i][j], q_t0_un[i][j], new_q_t0[i][j]), end='')
            print()

   
    def save_weights(self, n_ep):
        self.saver.save(self.sess, "./%s_%d" % (self.output_file, n_ep))

    def load_weights(self, filename):
        self.saver.restore(self.sess, filename)
