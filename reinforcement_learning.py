from ppo import PPO
from environment import Environment
from experience import Experience
from episode import Episode
from train import Train
from a2c import A2C


def train_episode(sess, actor, environment, train):
    env = environment.env
    observation = env.reset()
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

        if done:
            if train is True:
                actor.sync_target(sess)
                episode.set_terminal_state_value(actor.get_value(sess, observation))
                episode, global_step = actor.learn(sess, episode)
                return episode, global_step
            break


def init():
    env = Environment(action_space_length=2, observation_space_length=4, gym_string="CartPole-v0", max_step=300)
    actor = PPO(num_units=40,
                num_layers=2,
                action_space_len=env.action_space_length,
                observe_space_len=env.observation_space_length,
                activation="elu",
                discount_ratio=0.9,
                learning_rate=0.0001,
                n_step=2,
                clip_r=0.3
                )
    train = Train(is_train=True, max_episode=2500)
    return actor, env, train
