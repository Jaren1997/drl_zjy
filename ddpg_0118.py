# coding=utf-8
import numpy as np
import gym
import pandas as pd
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

GAMMA = 0.9     # reward discount in TD error
LR_A = 0.001    # learning rate for actor
LR_C = 0.01     # learning rate for critic
OUTPUT_GRAPH = False
MAX_EPISODE = 500
DISPLAY_REWARD_THRESHOLD = 200  # renders environment if total episode reward is greater then this threshold
MAX_EP_STEPS = 2000   # maximum time step in one episode
RENDER = False  # rendering wastes time

class Actor(object):
    def __init__(self, sess, act_dim, obs_dim, lr):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, obs_dim], "state")
        self.a = tf.placeholder(tf.float32, [1, act_dim], "action")
        self.Q = tf.placeholder(tf.float32, None, "Q")
        self.cost = self.Q = tf.placeholder(tf.float32, None, "cost")
        
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

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.cost)

        with tf.variable_scope('cost'):
            self.cost = tf.reduce_mean(-1.0 * self.Q)

    def choose_action(self, obs):
        # s = np.hstack((obs[0], obs[1]))[np.newaxis, :]
        # return self.sess.run(self.acts_prob, {self.s: s})
        s = obs[np.newaxis, :] # 外面再套一个[]
        probs = self.sess.run(self.acts_prob, {self.s: s})   # get probabilities for all actions
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())

    def learn(self, Q): # Q从critic网络得到
        self.sess.run(self.train_op, {self.cost: Q})
        
class Critic(object):
    def __init__(self, sess, act_dim, obs_dim, lr):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, obs_dim], "state")
        self.a = tf.placeholder(tf.float32, [1, act_dim], "action")
        self.s_a = tf.concat([self.s, self.a], 1)
        self.target_Q_next = tf.placeholder(tf.float32, None, "target_Q")
        self.r = tf.placeholder(tf.float32, None, "r")


        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(
                inputs=self.s_a,
                units=(obs_dim + act_dim) * 2,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.Q = tf.layers.dense(
                inputs=l1,
                units=1,
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='Q'
            )

        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.r + GAMMA * self.target_Q_next - self.Q
            self.loss = tf.square(self.td_error)

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, Q, r, target_Q_next, ):
        self.sess.run([self.td_error, self.train_op], {self.Q: Q, self.target_Q_next: target_Q_next, self.r: r})

    def value(self, obs, action):
        # s = np.hstack((obs[0], obs[1]))[np.newaxis, :]
        s = obs[np.newaxis, :] # 外面再套一个[]
        s_a = tf.concat([s, action], 1)
        self.sess.run(self.Q, {self.s_a: s_a})
        return self.Q

class DDPG(object):
    def __init__(self, sess, actor_model, critic_model):
        self.actor_model = actor_model
        self.critic_model = critic_model
        self.sess = sess

    def actor_learn(self, obs, action):
        Q = self.critic_model.value(obs, action)
        self.actor_model.learn(Q)
        
    def critic_learn(self, obs, action, reward, obs_next):
        Q = self.critic_model.value(obs, action)
        a_next = self.actor_model.choose_action(obs_next) # 这里应该用target_policy网络
        target_Q_next = self.critic_model.value(obs_next, a_next) # 这里应该用target_Q网络
        self.critic_model.learn(Q, reward, target_Q_next)

env = gym.make('CartPole-v0')
env.seed(1)
env = env.unwrapped

obs_dim = env.observation_space[0]
act_dim = env.action_space.n

sess = tf.Session()
actor = Actor(sess, act_dim = act_dim, obs_dim = obs_dim, lr=LR_A)
critic = Critic(sess, act_dim = act_dim, obs_dim = obs_dim, lr=LR_C)
ddpg = DDPG(sess, actor, critic)

sess.run(tf.global_variables_initializer())

res = []
for i_episode in range(MAX_EPISODE):
    s = env.reset()
    t = 0
    track_r = []
    while True:
        if RENDER: env.render()

        a = actor.choose_action(s)
        s_, r, done, info = env.step(a)
        if done: r = -20

        ddpg.actor_learn(s, a)
        ddpg.critic_learn(s, a, r, s_)

        s = s_
        t += 1

        if done or t >= MAX_EP_STEPS:
            ep_rs_sum = sum(track_r)

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
            if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True  # rendering
            print("episode:", i_episode, "  reward:", int(running_reward))
            res.append([i_episode, running_reward])
            break

pd.DataFrame(res,columns=['episode','a2c_reward']).to_csv('../a2c_reward.csv')
