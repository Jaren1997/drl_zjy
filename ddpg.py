# coding=utf-8
import numpy as np
from copy import deepcopy
import parl
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import math

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
    hid_size = 100 # 这个是啥
    self.fc1 = nn.Linear(obs_dim, hid_size)
    self.fc2 = nn.Linear(hid_size, act_dim)

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

class DDPG(parl.Algorithm): # 包含了对两个网络及其目标网络的更新步骤
  def __init__(self, model, gamma = None, tau = None, actor_lr = None, critic_lr = None):
    assert isinstance(gamma, float) # reward的衰减参数
    assert isinstance(tau, float) # self.target_model和self.model同步参数的软更新参数
    assert isinstance(actor_lr, float) # actor的学习率
    assert isinstance(critic_lr, float) # critic的学习率
    self.gamma = gamma
    self.tau = tau
    self.actor_lr = actor_lr
    self.critic_lr = critic_lr
    self.model = model
    self.target_model = deepcopy(model)

  def predict(self, obs):
    return self.model.policy(obs) # 输入actor网络中，得出在obs下需要采取的动作

  def learn(self, obs, action, reward, next_obs, terminal):
    # 更新actor和critic
    actor_cost = self._actor_learn(obs)
    critic_cost = self._critic_learn(obs, action, reward, next_obs, terminal)
    return actor_cost, critic_cost

  def _actor_learn(self, obs): # 如果是直接用经验池中的数据更新，这里是不是要改
    action = self.model.policy(obs) # 输入obs，获得action
    Q = self.model.value(obs, action) # 输入obs和action，获得在该obs下做出action的Q值
    cost = parl.layers.reduce_mean(-1.0 * Q) # 平均值
    optimizer = paddle.optimizer.AdamOptimizer(self.actor_lr)
    optimizer.minimize(cost, parameter_list = self.model.get_actor_params())
    return cost

  def _critic_learn(self, obs, action, reward, next_obs, terminal):
    next_action = self.target_model.policy(next_obs) # 用来计算Q_next用的
    next_Q = self.target_model.value(next_obs, next_action) # S_next的Q值
    terminal = parl.layers.cast(terminal, dtype='float32') # terminal是啥
    target_Q = reward + (1.0 - terminal) * self.gamma * next_Q # 相当于一个靶子
    target_Q.stop_gradient = True # 阻止更新网络参数

    Q = self.model.value(obs, action) # 这一次的Q
    cost = parl.layers.square_error_cost(Q, target_Q)
    cost = parl.layers.reduce_mean(cost)
    optimizer = paddle.optimizer.AdamOptimizer(self.critic_lr)
    optimizer.minimize(cost)
    return cost

  def sync_target(self, decay = None):
    # 更新target_Q网络
    if decay is None:
      decay = 1.0 - self.tau
      # 新参数占0.1%的权重，旧参数为99.9%的权重，使得参数更新更平滑。该方法称为软更新。
    self.model.sync_weights_to(self.target_model, decay = decay)

class Agent(parl.Agent):
  def __init__(self, algorithm, obs_dim, act_dim):
    assert isinstance(obs_dim, int)
    assert isinstance(act_dim, int)
    self.obs_dim = obs_dim
    self.act_dim = act_dim

    super(parl.Agent, self).__init__(algorithm)

    # 最开始先同步self.model和self.target_model的参数
    self.alg.sync_target(decay = 0)

  def build_program(self):
    self.pred_program = paddle.static.Program()
    self.learn_program = paddle.static.Program()

    with paddle.static.program_guard(self.pred_program):
      # 预测程序
      # 输入的参数
      obs = parl.layers.data(name = 'obs', shape=[self.obs_dim], dtype='float32')
      # 输出的参数
      self.pred_act = self.alg.predict(obs)

    with paddle.static.program_guard(self.learn_program):
      # 学习程序
      # 输入的参数
      obs = parl.layers.data(name = 'obs', shape = [self.obs_dim], dtype = 'float32')
      act = parl.layers.data(name = 'act', shape=[self.act_dim], dtype = 'float32')
      reward = parl.layers.data(name = 'reward', shape=[], dtype='float32')
      next_obs = parl.layers.data(name='next_obs', shape=[self.obs_dim], dtype='float32')
      terminal = parl.layers.data(name='terminal', shape=[], dtype='bool') # episode是否终止的标志。这里，我们的程序是否可以设计成，某一条队的队伍超过一定长度就视作无法完成了，视作终止状态？
      # 输出的参数
      _, self.critic_cost = self.alg.learn(obs, act, reward, next_obs, terminal)

  def predict(self, obs):
    obs = np.expand_dims(obs, axis = 0)
    act = self.fluid_executor.run(self.pred_program, feed={'obs': obs}, fetch_list = [self.pred_act])[0] # 运行predict程序，输出act
    act = parl.layers.squeeze(act)
    return act

  def learn(self, obs, act, reward, next_obs, terminal):
    feed = {
      'obs': obs,
      'act': act,
      'reward': reward,
      'next_obs': next_obs,
      'terminal': terminal
    }
    critic_cost = self.fluid_executor.run(self.learn_program, feed = feed, fetch_list = [self.critic_cost])[0] # 运行learn程序
    self.alg.sync_target()
    return critic_cost

class Compare_agent():
  # 7个对比agent
  def __init__(self, env):
    self.agents = []
    self.action_compared = np.zeros(shape=(env.num_server - 1, env.num_server + 1))

  def react(self, obs, env):
    for index in range(len(self.agents)):
      # 多余的平均分配到离自己最近的服务器上进行计算
      if self.agents[index] == 'nearest':
        self.action_compared[index][index + 1] = min(max(env.compute_ability - obs[1][index], 0), obs[1][index])
        remain = obs[0][index] - self.action_compared[index][index + 1]
        if remain > 0:
          # 如果本地资源用尽了还有未完成的任务，就随机分配到离自己最近的服务器上进行计算
          neighbor_map = env.neighbor_map[index + 1]
          if len(neighbor_map) == 1:
            self.action_compared[index][neighbor_map[index + 1][0]] = remain
          else:
            sum_neighbor = 0
            for i in neighbor_map:
              self.action_compared[index][i] = math.floor(remain / len(neighbor_map))
              sum_neighbor += self.action_compared[index][i]
            if sum_neighbor < remain:
              self.action_compared[index][neighbor_map[0]] += (remain - sum_neighbor)
      
      # 多余的平均分配到所有服务器上进行计算
      if self.agents[index] == 'average':
        self.action_compared[index][index + 1] = min(max(env.compute_ability - obs[1][index], 0), obs[1][index])
        remain = obs[0][index] - self.action_compared[index][index + 1]
        if remain > 0:
          sum_temp = 0
          for i in range(1, env.num_server + 1):
            if self.action_compared[index][i] == 0:
              self.action_compared[index][i] = math.floor(remain / (env.num_server - 1))
              sum_temp += self.action_compared[index][i]
          if sum_temp < remain:
            self.action_compared[index][np.random.randint(1,8)] += remain - sum_temp
      
      # 所有都在本地计算
      if self.agents[index] == 'local' :
        self.action_compared[index][index + 1] = obs[0][index]

      # 多余的传递到云上计算
      if self.agents[index] == 'cloud' : # 把这个agent的所有任务放到cloud上。
        self.action_compared[index][index + 1] = min(max(env.compute_ability - obs[1][index], 0), obs[1][index])
        remain = obs[0][index] - self.action_compared[index][index + 1]
        if remain > 0:
          self.action_compared[index][0] = remain
      
      # 多余的传递到本时隙初最短的队伍上计算
      if self.agents[index] == 'fastest' :
        self.action_compared[index][index + 1] = min(max(env.compute_ability - obs[1][index], 0), obs[1][index])
        remain = obs[0][index] - self.action_compared[index][index + 1]
        if remain > 0:
          sum_temp = 0
          min_idxs = [idx + 1 for idx, val in enumerate(obs[1]) if val == np.min(obs[1])]
          for i in min_idxs:
            self.action_compared[index][i + 1] += math.floor(remain / (env.num_server - 1))
            sum_temp += self.action_compared[index][i]
          if sum_temp < remain:
            self.action_compared[index][min_idxs[0]] += remain - sum_temp

      return self.action_compared