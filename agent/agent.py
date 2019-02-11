from common.environment import Environment
from common.experience import Experience
from common.episode import Episode
from common.train import Train
from feature.gae import GAE
from model.feed_forward import Model
from algorithum.ppo import PPO
import parameter.parameters as ps


def train_episode(sess, actor, environment, train):
    env = environment.env
    observation = env.reset()

    entire_episode = Episode()
    episode = Episode()

    for step in range(train.max_step):

        if train.train is False:
            env.render()

        last_state_observation = observation

        action, value = actor.choose_action(sess, observation)
        observation, reward, done, info = env.step(action)

        exp = Experience()
        exp.set_all(reward=reward, action=action, last_state_obs=last_state_observation, current_state_obs=observation, last_state_value=value)
        '''
        (reward + 8) / 8)
        '''
        episode.add_experience(exp)
        episode.add_reward(reward)

        if done or train.stop_to_learn(current_step=step):

            if observation is None:
                episode.set_terminal_state_value(0)
            else:
                episode.set_terminal_state_value(actor.get_value(sess, observation))

            if train.train is True:
                episode = actor.learn(sess, episode)

            entire_episode.add_episode(episode)
            episode = Episode()
        if done:
            break
    return entire_episode


def init():
    return ps.get_racingPPO_LSTM()
