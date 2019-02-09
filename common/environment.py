import gym
from gym import wrappers


class Environment:

    def __init__(self, discrete_action_bound, observation_space_dimension, action_space_dimension, gym_string,
                 is_continuous):
        self.discrete_action_bound = discrete_action_bound
        self.observation_space_dimension = observation_space_dimension
        self.action_space_dimension = action_space_dimension
        self.gym_string = gym_string
        self.is_continuous = is_continuous
        self.a_bound = None
        self.env = self.make_env()

    def make_env(self):
        env = gym.make(self.gym_string)
        if self.is_continuous is True:
            self.a_bound = [env.action_space.low, env.action_space.high]
        else:
            self.a_bound = None
        return env
