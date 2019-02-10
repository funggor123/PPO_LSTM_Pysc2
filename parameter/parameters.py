from common.environment import Environment
from common.train import Train
from feature.gae import GAE
from model.feed_forward import Model
from algorithum.ppo import PPO


def get_car_pole():
    env = Environment(discrete_action_bound=[2], observation_space_dimension=(4,), action_space_dimension=(1,),
                      is_continuous=False, gym_string="CartPole-v0")
    feature_transform = GAE(obs_dimension=env.observation_space_dimension, a_dimension=env.action_space_dimension,
                            gamma=0.99, beta=1, is_continuous=env.is_continuous)
    feed_forward = Model(a_len=env.discrete_action_bound, a_dimension=env.action_space_dimension,
                         obs_dimension=env.observation_space_dimension, is_continuous=env.is_continuous,
                         a_bound=env.a_bound)
    actor = PPO(obs_dimension=env.observation_space_dimension, lr=0.0001, feature_transform=feature_transform,
                model=feed_forward,
                epsilon=0.1,
                a_dimension=env.action_space_dimension,
                action_space_length=env.discrete_action_bound,
                regular_str=0.01,
                )
    train = Train(train=True, max_episode=5e5, max_step=400, batch_size=200)
    return actor, env, train


def get_pendulum():
    env = Environment(discrete_action_bound=[2], observation_space_dimension=(3,), action_space_dimension=(1,),
                      is_continuous=True, gym_string="Pendulum-v0")
    feature_transform = GAE(obs_dimension=env.observation_space_dimension, a_dimension=env.action_space_dimension,
                            gamma=0.90, beta=1, is_continuous=env.is_continuous)
    feed_forward = Model(a_len=env.discrete_action_bound, a_dimension=env.action_space_dimension,
                         obs_dimension=env.observation_space_dimension, is_continuous=env.is_continuous,
                         a_bound=env.a_bound)
    actor = PPO(obs_dimension=env.observation_space_dimension, lr=0.0001, feature_transform=feature_transform,
                model=feed_forward,
                epsilon=0.1,
                a_dimension=env.action_space_dimension,
                action_space_length=env.discrete_action_bound,
                regular_str=1e-2,
                )
    train = Train(train=True, max_episode=5e5, max_step=200, batch_size=32)
    return actor, env, train

