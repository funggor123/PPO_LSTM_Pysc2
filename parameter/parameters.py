from common.environment import Environment
from common.train import Train
from feature.gae import GAE
from model.feed_forward import Model
from algorithum.ppo import PPO
from model.cnn import ConvNet
from algorithum.a2c import A2C
from model.conv_lstm import ConvLSTM


def get_car_pole(worker=None):
    env = Environment(discrete_action_bound=[2], observation_space_dimension=(4,), action_space_dimension=(1,),
                      is_continuous=False, gym_string="CartPole-v0")
    feature_transform = GAE(obs_dimension=env.observation_space_dimension, a_dimension=env.action_space_dimension,
                            gamma=0.99, beta=1, is_continuous=env.is_continuous, max_reward=1, min_reward=0)
    feed_forward = Model(a_len=env.discrete_action_bound, a_dimension=env.action_space_dimension,
                         obs_dimension=env.observation_space_dimension, is_continuous=env.is_continuous,
                         a_bound=env.a_bound, is_cat=True)
    actor = PPO(obs_dimension=env.observation_space_dimension, lr=1e-4, feature_transform=feature_transform,
                model=feed_forward,
                epsilon=0.1,
                a_dimension=env.action_space_dimension,
                action_space_length=env.discrete_action_bound,
                regular_str=1e-2,
                worker=worker,
                minibatch=16,
                vf_coef=1,
                epoch=5,
                max_grad_norm=0.5,
                )
    train = Train(train=True, max_episode=1e5, max_step=10000, batch_size=64)
    return actor, env, train


def get_pendulumPPO(worker=None):
    env = Environment(discrete_action_bound=[2], observation_space_dimension=(3,), action_space_dimension=(1,),
                      is_continuous=True, gym_string="Pendulum-v0", worker=worker)
    feature_transform = GAE(obs_dimension=env.observation_space_dimension, a_dimension=env.action_space_dimension,
                            gamma=0.90, beta=0.95, is_continuous=env.is_continuous, max_reward=0, min_reward=-16.2736044)
    feed_forward = Model(a_len=env.discrete_action_bound, a_dimension=env.action_space_dimension,
                         obs_dimension=env.observation_space_dimension, is_continuous=env.is_continuous,
                         a_bound=env.a_bound)
    actor = PPO(obs_dimension=env.observation_space_dimension, lr=0.0001, feature_transform=feature_transform,
                model=feed_forward,
                a_dimension=env.action_space_dimension,
                action_space_length=env.discrete_action_bound,
                regular_str=1e-2,
                worker=worker,
                epsilon=0.1,
                vf_coef=1,
                max_grad_norm=0.5,
                minibatch=32,
                epoch=10
                )
    train = Train(train=True, max_episode=5e5, max_step=10000, batch_size=8192, print_every_episode=100)
    return actor, env, train


def get_pendulumA2C(worker=None):
    env = Environment(discrete_action_bound=[2], observation_space_dimension=(3,), action_space_dimension=(1,),
                      is_continuous=True, gym_string="Pendulum-v0")
    feature_transform = GAE(obs_dimension=env.observation_space_dimension, a_dimension=env.action_space_dimension,
                            gamma=0.90, beta=1, is_continuous=env.is_continuous, max_reward=0, min_reward=-16.2736044)
    feed_forward = Model(a_len=env.discrete_action_bound, a_dimension=env.action_space_dimension,
                         obs_dimension=env.observation_space_dimension, is_continuous=env.is_continuous,
                         a_bound=env.a_bound)
    actor = A2C(obs_dimension=env.observation_space_dimension, lr=0.0001, feature_transform=feature_transform,
                model=feed_forward,
                a_dimension=env.action_space_dimension,
                action_space_length=env.discrete_action_bound,
                regular_str=1e-2,
                minibatch=32,
                epoch=10,
                max_grad_norm=0.5,
                )
    train = Train(train=True, max_episode=5e5, max_step=10000, batch_size=8192)
    return actor, env, train


def get_racingPPO_CNN(worker=None):
    env = Environment(discrete_action_bound=[2], observation_space_dimension=(96, 96, 3), action_space_dimension=(3,),
                      is_continuous=True, gym_string="CarRacing-v0")
    feature_transform = GAE(obs_dimension=env.observation_space_dimension, a_dimension=env.action_space_dimension,
                            gamma=0.99, beta=0.95, is_continuous=env.is_continuous, max_reward=100, min_reward=-0.10000000000000142)
    conv = ConvNet(a_len=env.discrete_action_bound, a_dimension=env.action_space_dimension,
                           obs_dimension=env.observation_space_dimension, is_continuous=env.is_continuous,
                           a_bound=env.a_bound)
    actor = PPO(obs_dimension=env.observation_space_dimension, lr=0.0001, feature_transform=feature_transform,
                model=conv,
                worker=worker,
                epsilon=0.1,
                a_dimension=env.action_space_dimension,
                action_space_length=env.discrete_action_bound,
                regular_str=1e-2,
                minibatch=128,
                vf_coef=1,
                epoch=10,
                max_grad_norm=0.5,
                )
    train = Train(train=True, max_episode=5e5, max_step=100000, batch_size=8192, print_every_episode=1)
    return actor, env, train


def get_racingPPO_LSTM(worker=None):
    env = Environment(discrete_action_bound=[2], observation_space_dimension=(96, 96, 3), action_space_dimension=(3,),
                      is_continuous=True, gym_string="CarRacing-v0")
    feature_transform = GAE(obs_dimension=env.observation_space_dimension, a_dimension=env.action_space_dimension,
                            gamma=0.90, beta=0.95, is_continuous=env.is_continuous, max_reward=1, min_reward=0)
    feed_forward = ConvLSTM(a_len=env.discrete_action_bound, a_dimension=env.action_space_dimension,
                            obs_dimension=env.observation_space_dimension, is_continuous=env.is_continuous,
                            a_bound=env.a_bound)
    actor = PPO(obs_dimension=env.observation_space_dimension, lr=1e-3, feature_transform=feature_transform,
                model=feed_forward,
                epsilon=0.1,
                worker=worker,
                a_dimension=env.action_space_dimension,
                action_space_length=env.discrete_action_bound,
                regular_str=1e-2,
                minibatch=8,
                epoch=10,
                max_grad_norm=0.5,
                vf_coef=1
                )
    train = Train(train=True, max_episode=5e5, max_step=512, batch_size=128, print_every_episode=1)
    return actor, env, train
