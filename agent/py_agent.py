from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from absl import app
from feature import py_feature
from feature.py_gae import PY_GAE
from model.pysc2_conv_net import ConvNet
from algorithum.py_a2c import Py_A2C
import tensorflow as tf
from common.episode import Episode
import random


class ZergAgent(base_agent.BaseAgent):

    def __init__(self, actor, sess):
        super(ZergAgent, self).__init__()
        self.actor = actor
        self.sess = sess
        self.v = None

    def step(self, obs):
        super(ZergAgent, self).step(obs)
        act_id, act_args, self.v = self.actor.choose_action(self.sess, obs)
        return actions.FunctionCall(act_id, act_args)


def main(unused_argv):
    max_step = 1e6
    lr = 5e-4
    max_agent_step = 64
    with tf.Session() as sess:
        gae = PY_GAE(ssize=64, msize=64, gamma=0.99, beta=1)
        convnet = ConvNet()
        actor = Py_A2C(
            lr=0.0001,
            msize=64,
            ssize=64,
            feature_transform=gae,
            model=convnet,
            regular_str=1e-2,
            minibatch=32,
            epoch=10)
        agent = ZergAgent(actor, sess)
        sess.run(actor.get_init_opr())
        try:
            with sc2_env.SC2Env(
                    map_name="MoveToBeacon",
                    players=[sc2_env.Agent(sc2_env.Race.zerg)],
                    agent_interface_format=features.AgentInterfaceFormat(
                        feature_dimensions=sc2_env.Dimensions(screen=64, minimap=64),
                        use_feature_units=True),
                    step_mul=8,
                    game_steps_per_episode=0,
                    visualize=True) as env:
                num = 0
                global_step = 0
                while True:

                    agent.setup(env.observation_spec(), env.action_spec())

                    timesteps = env.reset()
                    agent.reset()
                    rbs = []
                    episode = Episode()
                    ep_step = 0
                    while True:
                        last_timesteps = timesteps
                        step_actions = [agent.step(timesteps[0])]
                        if timesteps[0].last() or ep_step >= max_agent_step:
                            print(timesteps[0].reward)
                            learning = lr * (1 - 0.9 * global_step / max_step)
                            actor.learn(sess, rbs, learning)
                            print("----reward----")
                            print("reward: ", episode.reward)
                            print("global_step", global_step)
                            print("episode: ", num)
                            num = num + 1
                            global_step = ep_step + global_step
                            print("----reward----")
                            break
                        timesteps = env.step(step_actions)
                        episode.add_reward(timesteps[0].reward)
                        ep_step = ep_step + 1
                        rbs.append([last_timesteps[0], step_actions[0], timesteps[0], agent.v])

        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    app.run(main)
