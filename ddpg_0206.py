import numpy as np
from copy import deepcopy
import parl
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class Model(parl.Model): # 包括了两个model，包括了actor和critic网络
  def __init__(self, act_dim, obs_dim):
    super(Model, self).__init__()
    self.actor_model = ActorModel(act_dim, obs_dim)
    self.critic_model = CriticModel(act_dim, obs_dim)

  def policy(self, obs):
    return self.actor_model.policy(obs)

  def value(self, obs, act):
    return self.critic_model.value(obs, act)

  def get_actor_params(self):
    return self.actor_model.parameters()

class ActorModel(parl.Model): # 演员模型
  def __init__(self, act_dim, obs_dim):
    super(ActorModel, self).__init__()
    
    self.s = tf.placeholder(tf.float32, [1, obs_dim], "state")
    self.a = tf.placeholder(tf.float32, [1, act_dim], "action")
    self.Q = tf.placeholder(tf.float32, None, "Q")
    
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

  def policy(self, obs):
    # hid = self.fc1(obs) # hid为第一层的输出
    # means = self.fc2(hid) # 第二层采用第一层的输出为输入，输出means，为一个-1到1之间的浮点数；这里要改；
    hid = F.relu(self.fc1(obs))
    means = F.tanh(self.fc2(hid))
    return means

class CriticModel(parl.Model): # 评论家模型
  def __init__(self, act_dim, obs_dim):
    super(CriticModel, self).__init__()
    hid_size = 100
    # self.fc1 = layers.fc(size = hid_size, act='relu')
    # self.fc2 = layers.fc(size = 1, act=None) # 评论家模型输出的是Q值，不需要激活函数
    self.fc1 = nn.Linear(act_dim + obs_dim, hid_size)
    self.fc2 = nn.Linear(hid_size, act_dim)
    
  def value(self, obs, act): # 输入有agent的观察obs以及采取的动作act
    concat = parl.layers.concat([obs, act], axis = 1) # 把输入的obs和act做了拼接，沿着第二个维度进行拼接，即行数不变，列数增加
    # hid = self.fc1(concat)
    # Q = self.fc2(hid)
    hid = F.relu(self.fc1(concat))
    Q = self.fc2(hid, 1)
    Q = parl.layers.squeeze(Q, axes = [1]) # 压缩一维数据 删除第二列的数据？
    return Q