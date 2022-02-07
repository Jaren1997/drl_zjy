import parl
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from copy import deepcopy
from parl.utils.utils import check_model_method
import numpy as np

class Model(parl.Model):
    def __init__(self, obs_dim, action_dim):
        super(Model, self).__init__()
        self.actor_model = Actor(obs_dim, action_dim)
        self.critic_model = Critic(obs_dim, action_dim)

    def policy(self, obs):
        return self.actor_model(obs)

    def value(self, obs, action):
        return self.critic_model(obs, action)

    def get_actor_params(self):
        return self.actor_model.parameters()

    def get_critic_params(self):
        return self.critic_model.parameters()

class Actor(parl.Model):
    def __init__(self, obs_dim, action_dim):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(obs_dim, 2 * obs_dim)
        self.l2 = nn.Linear(2 * obs_dim, 2 * action_dim)
        self.l3 = nn.Linear(2 * action_dim, action_dim)

    def forward(self, obs): # 这个是啥呢
        a = F.relu(self.l1(obs))
        a = F.relu(self.l2(a))
        return self.l3(a) # 直接返回一个1*action_dim的向量


class Critic(parl.Model):
    def __init__(self, obs_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(obs_dim, 400)
        self.l2 = nn.Linear(400 + action_dim, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, obs, action): # 把action放到了中间层
        q = F.relu(self.l1(obs))
        q = F.relu(self.l2(paddle.concat([q, action], 1)))
        return self.l3(q)

class DDPG(parl.Algorithm):
    def __init__(self,
                 model,
                 gamma=None,
                 tau=None,
                 actor_lr=None,
                 critic_lr=None):
        """ DDPG algorithm

        Args:
            model(parl.Model): forward network of actor and critic.
            gamma(float): discounted factor for reward computation
            tau (float): decay coefficient when updating the weights of self.target_model with self.model
            actor_lr (float): learning rate of the actor model
            critic_lr (float): learning rate of the critic model
        """
        # checks
        check_model_method(model, 'value', self.__class__.__name__)
        check_model_method(model, 'policy', self.__class__.__name__)
        check_model_method(model, 'get_actor_params', self.__class__.__name__)
        check_model_method(model, 'get_critic_params', self.__class__.__name__)
        assert isinstance(gamma, float)
        assert isinstance(tau, float)
        assert isinstance(actor_lr, float)
        assert isinstance(critic_lr, float)

        self.gamma = gamma
        self.tau = tau
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        self.model = model
        self.target_model = deepcopy(self.model)
        self.actor_optimizer = paddle.optimizer.Adam(
            learning_rate=actor_lr, parameters=self.model.get_actor_params())
        self.critic_optimizer = paddle.optimizer.Adam(
            learning_rate=critic_lr, parameters=self.model.get_critic_params())

    def predict(self, obs):
        return self.model.policy(obs)

    def learn(self, obs, action, reward, next_obs, terminal):
        critic_loss = self._critic_learn(obs, action, reward, next_obs,
                                         terminal)
        actor_loss = self._actor_learn(obs)

        self.sync_target()
        return critic_loss, actor_loss

    def _critic_learn(self, obs, action, reward, next_obs, terminal):
        with paddle.no_grad():
            # Compute the target Q value
            target_Q = self.target_model.value(
                next_obs, self.target_model.policy(next_obs))
            terminal = paddle.cast(terminal, dtype='float32')
            target_Q = reward + ((1. - terminal) * self.gamma * target_Q)

        # Get current Q estimate
        current_Q = self.model.value(obs, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Optimize the critic
        self.critic_optimizer.clear_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        return critic_loss

    def _actor_learn(self, obs):
        # Compute actor loss and Update the frozen target models
        actor_loss = -self.model.value(obs, self.model.policy(obs)).mean()

        # Optimize the actor
        self.actor_optimizer.clear_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        return actor_loss

    def sync_target(self, decay=None):
        """ update the target network with the training network

        Args:
            decay(float): the decaying factor while updating the target network with the training network.
                        0 represents the **assignment**. None represents updating the target network slowly that depends on the hyperparameter `tau`.
        """
        if decay is None:
            decay = 1.0 - self.tau
        self.model.sync_weights_to(self.target_model, decay=decay)

class Agent(parl.Agent):
    def __init__(self, algorithm, act_dim, expl_noise=0.1):
        assert isinstance(act_dim, int)
        super(Agent, self).__init__(algorithm)

        self.act_dim = act_dim
        self.expl_noise = expl_noise

        self.alg.sync_target(decay=0)

    def sample(self, obs):
        action_numpy = self.predict(obs)
        action_noise = np.random.normal(0, self.expl_noise, size=self.act_dim)
        action = (action_numpy + action_noise).clip(-1, 1)
        return action

    def predict(self, obs):
        obs = paddle.to_tensor(obs.reshape(1, -1), dtype='float32')
        action = self.alg.predict(obs)
        action_numpy = action.cpu().numpy()[0]
        return action_numpy

    def learn(self, obs, action, reward, next_obs, terminal):
        terminal = np.expand_dims(terminal, -1)
        reward = np.expand_dims(reward, -1)

        obs = paddle.to_tensor(obs, dtype='float32')
        action = paddle.to_tensor(action, dtype='float32')
        reward = paddle.to_tensor(reward, dtype='float32')
        next_obs = paddle.to_tensor(next_obs, dtype='float32')
        terminal = paddle.to_tensor(terminal, dtype='float32')
        critic_loss, actor_loss = self.alg.learn(obs, action, reward, next_obs,
                                                 terminal)
        return critic_loss, actor_loss