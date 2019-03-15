from common.environment import Environment
from common.train import Train
from feature.gae import GAE
from model.feed_forward import Model
from algorithum.ppo import PPO
from model.conv_net import ConvNet
from algorithum.a2c import A2C
from model.conv_lstm import ConvLSTM


def get_car_pole():
    env = Environment(discrete_action_bound=[2], observation_space_dimension=(4,), action_space_dimension=(1,),
                      is_continuous=False, gym_string="CartPole-v0")
    feature_transform = GAE(obs_dimension=env.observation_space_dimension, a_dimension=env.action_space_dimension,
                            gamma=0.99, beta=1, is_continuous=env.is_continuous)
    feed_forward = Model(a_len=env.discrete_action_bound, a_dimension=env.action_space_dimension,
                         obs_dimension=env.observation_space_dimension, is_continuous=env.is_continuous,
                         a_bound=env.a_bound, is_cat=False)
    actor = PPO(obs_dimension=env.observation_space_dimension, lr=0.0001, feature_transform=feature_transform,
                model=feed_forward,
                epsilon=0.15,
                a_dimension=env.action_space_dimension,
                action_space_length=env.discrete_action_bound,
                regular_str=0.01,
                minibatch=16,
                vf_coef=0.5,
                epoch=3,
                max_grad_norm=0.5,
                is_seperate=False
                )
    train = Train(train=True, max_episode=5e5, max_step=10000, batch_size=128)
    return actor, env, train


def get_pendulumPPO():
    env = Environment(discrete_action_bound=[2], observation_space_dimension=(3,), action_space_dimension=(1,),
                      is_continuous=True, gym_string="Pendulum-v0")
    feature_transform = GAE(obs_dimension=env.observation_space_dimension, a_dimension=env.action_space_dimension,
                            gamma=0.90, beta=1, is_continuous=env.is_continuous)
    feed_forward = Model(a_len=env.discrete_action_bound, a_dimension=env.action_space_dimension,
                         obs_dimension=env.observation_space_dimension, is_continuous=env.is_continuous,
                         a_bound=env.a_bound)
    actor = PPO(obs_dimension=env.observation_space_dimension, lr=0.0001, feature_transform=feature_transform,
                model=feed_forward,
                a_dimension=env.action_space_dimension,
                action_space_length=env.discrete_action_bound,
                regular_str=1e-2,
                epsilon=0.1,
                vf_coef=1,
                max_grad_norm=0.5,
                minibatch=32,
                epoch=10
                )
    train = Train(train=True, max_episode=5e5, max_step=10000, batch_size=8192)
    return actor, env, train


def get_pendulumA2C():
    env = Environment(discrete_action_bound=[2], observation_space_dimension=(3,), action_space_dimension=(1,),
                      is_continuous=True, gym_string="Pendulum-v0")
    feature_transform = GAE(obs_dimension=env.observation_space_dimension, a_dimension=env.action_space_dimension,
                            gamma=0.90, beta=1, is_continuous=env.is_continuous)
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


def get_racingPPO_CNN():
    env = Environment(discrete_action_bound=[2], observation_space_dimension=(96, 96, 3), action_space_dimension=(3,),
                      is_continuous=True, gym_string="CarRacing-v0")
    feature_transform = GAE(obs_dimension=env.observation_space_dimension, a_dimension=env.action_space_dimension,
                            gamma=0.90, beta=1, is_continuous=env.is_continuous)
    feed_forward = ConvNet(a_len=env.discrete_action_bound, a_dimension=env.action_space_dimension,
                           obs_dimension=env.observation_space_dimension, is_continuous=env.is_continuous,
                           a_bound=env.a_bound)
    actor = PPO(obs_dimension=env.observation_space_dimension, lr=0.0001, feature_transform=feature_transform,
                model=feed_forward,
                epsilon=0.1,
                a_dimension=env.action_space_dimension,
                action_space_length=env.discrete_action_bound,
                regular_str=1e-2,
                minibatch=32,
                vf_coef=0.5,
                epoch=10,
                max_grad_norm=0.5,
                )
    train = Train(train=True, max_episode=5e5, max_step=200, batch_size=32, print_every_episode=1)
    return actor, env, train


def get_racingPPO_LSTM():
    env = Environment(discrete_action_bound=[2], observation_space_dimension=(96, 96, 3), action_space_dimension=(3,),
                      is_continuous=True, gym_string="CarRacing-v0")
    feature_transform = GAE(obs_dimension=env.observation_space_dimension, a_dimension=env.action_space_dimension,
                            gamma=0.90, beta=1, is_continuous=env.is_continuous)
    feed_forward = ConvLSTM(a_len=env.discrete_action_bound, a_dimension=env.action_space_dimension,
                            obs_dimension=env.observation_space_dimension, is_continuous=env.is_continuous,
                            a_bound=env.a_bound)
    actor = PPO(obs_dimension=env.observation_space_dimension, lr=0.0001, feature_transform=feature_transform,
                model=feed_forward,
                epsilon=0.1,
                a_dimension=env.action_space_dimension,
                action_space_length=env.discrete_action_bound,
                regular_str=1e-2,
                minibatch=32,
                epoch=10,
                max_grad_norm=0.5,
                vf_coef=1
                )
    train = Train(train=True, max_episode=5e5, max_step=200, batch_size=32, print_every_episode=1)
    return actor, env, train
