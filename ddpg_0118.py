# coding=utf-8
import numpy as np


import math
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class Actor(object):
    def __init__(self, sess, act_dim, obs_dim):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, obs_dim], "state")
        self.a = tf.placeholder(tf.float32, [1, act_dim], "action")
        
        with tf.variable_scope('Actor'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=2*obs_dim,    # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.acts_prob = tf.layers.dense(
                inputs=l1,
                units=act_dim,    # output units
                activation=tf.nn.softmax,   # get action probabilities
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='acts_prob'
            )

    def choose_action(self, s):
        s = np.hstack((s[0], s[1]))[np.newaxis, :]
        return self.sess.run(self.acts_prob, {self.s: s})

    def learn(self, s, a, Q):
        s = np.hstack((s[0], s[1]))[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.Q: Q}
        
