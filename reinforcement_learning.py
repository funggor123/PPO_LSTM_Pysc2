from ppo import PPO
from environment import Environment
from experience import Experience
from episode import Episode
from train import Train
from gae import GAE
from conv_net import ConvNet
from a2c import A2C


def train_episode(sess, actor, environment, train):
    env = environment.env
    observation = env.reset()
    entire_episode = Episode()
    episode = Episode()
    for ind in range(environment.max_step):

        if train is False:
            env.render()

        exp = Experience(obs_len=environment.observation_space_length, act_len=environment.action_space_length)
        exp.set_last_state(observation)

        if train is False:
            action, value = actor.choose_action(sess, observation)
        else:
            action, value = actor.choose_action_with_exploration(sess, observation)

        observation, reward, done, info = env.step(action)
        exp.set_action(action)
        exp.set_reward(reward)
        exp.set_last_state_value(value)
        exp.set_current_state(observation)

        episode.add_experience(exp)
        episode.add_reward(reward)
        if done or (train is True and ind % environment.batch_size == 0 and ind is not 0):
            global_step = 0
            if train is True:
                '''
                actor.sync_target(sess)
                '''
                if observation is None:
                    episode.set_terminal_state_value(0)
                else:
                    episode.set_terminal_state_value(actor.get_value(sess, observation))
                episode, global_step = actor.learn(sess, episode)
                entire_episode.add_reward(episode.acc_reward)
                entire_episode.add_loss(episode.loss)
                episode = Episode()
            if done:
                return entire_episode, global_step


def init():
    env = Environment(action_space_length=3, observation_space_length=(96, 96, 3), gym_string="CarRacing-v0",
                      max_step=2000, batch_size=300)
    gae = GAE(env.observation_space_length, env.action_space_length, discount_rate=0.9, n_step_rate=0.6)
    conv_net = ConvNet(activation='elu', a_len=env.action_space_length, o_len=env.observation_space_length,
                       is_continuous=True)
    actor = A2C(action_space_len=env.action_space_length,
                observe_space_len=env.observation_space_length,
                learning_rate=0.00001,
                feature_transform=gae,
                model=conv_net)
    train = Train(is_train=True, max_episode=2000)
    return actor, env, train
