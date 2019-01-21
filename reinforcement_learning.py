from a2c import A2C
from environment import Environment
from experience import Experience
from episode import Episode
from train import Train


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

        exp.set_last_state_value(value)
        exp.set_action(action)
        exp.set_reward(reward)
        exp.set_current_state(observation)

        episode.add_experience(exp)
        episode.add_reward(reward)
        episode.add_step()

        if done:
            if train is True:
                loss, global_step = actor.learn(sess, episode)
                return episode.reward, loss, global_step
            break


def init():
    env = Environment(action_space_length=2, observation_space_length=4, gym_string="CartPole-v0", max_step=300)
    actor = A2C(num_units=30,
                num_layers=2,
                action_space_len=env.action_space_length,
                observe_space_len=env.observation_space_length,
                activation="elu",
                discount_ratio=0.90,
                learning_rate=0.0005)
    train = Train(is_train=True, max_episode=2000)
    return actor, env, train
