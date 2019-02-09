from environment import Environment
from experience import Experience
from episode import Episode
from train import Train
from gae import GAE
from model import Model
from ppo import PPO
from a2c import A2C


def train_episode(sess, actor, environment, train):
    env = environment.env
    observation = env.reset()

    entire_episode = Episode()
    global_step = 0
    episode = Episode()

    for ind in range(environment.max_step):

        if train is False:
            env.render()

        exp = Experience(obs_len=environment.observation_space_length, act_len=environment.action_space_length)
        exp.set_last_state(observation)

        action, value = actor.choose_action(sess, observation)
        observation, reward, done, info = env.step(action)
        exp.set_action(action)
        exp.set_reward((reward + 8) / 8)
        exp.set_last_state_value(value)
        exp.set_current_state(observation)

        episode.add_experience(exp)
        episode.add_reward(reward)
        if ind == environment.max_step - 1 or done or ((ind % environment.batch_size) == 0 and ind is not 0):
            if observation is None:
                episode.set_terminal_state_value(0)
            else:
                episode.set_terminal_state_value(actor.get_value(sess, observation))
            if train is True:
                episode, global_step = actor.learn(sess, episode)
            entire_episode.add_reward(episode.reward)
            entire_episode.set_loss(episode.loss)
            episode = Episode()
        if done or ind == environment.max_step - 1:
            break
    return entire_episode, global_step


def init():
    env = Environment(action_space_length=[2], observation_space_length=(3,), gym_string="Pendulum-v0",
                      max_step=200, batch_size=32, action_dim=(1,), is_continuous=True)
    gae = GAE(env.observation_space_length, env.action_space_length, action_dim=env.action_dim, discount_rate=0.90,
              n_step_rate=1, is_continous=env.is_continuous)
    fn = Model(activation='elu', a_len=env.action_space_length, o_len=env.observation_space_length,
               is_continuous=env.is_continuous, a_bound=env.a_bound, a_dim=env.action_dim, num_layers=1, num_units=100)
    actor = PPO(observe_space_len=env.observation_space_length,
                learning_rate=0.0001,
                feature_transform=gae,
                model=fn,
                clip_r=0.1,
                action_space_dim=env.action_dim,
                action_space_length=env.action_space_length,
                regularization_stength=0.01
                )
    train = Train(is_train=True, max_episode=5e5)
    return actor, env, train
