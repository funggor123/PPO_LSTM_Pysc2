import gym
from gym import wrappers


class Environment:

    def __init__(self, action_space_length, observation_space_length, action_dim, gym_string, max_step, batch_size,
                 is_continuous):
        self.action_space_length = action_space_length
        self.observation_space_length = observation_space_length
        self.action_dim = action_dim
        self.gym_string = gym_string
        self.max_step = max_step
        self.env = gym.make(self.gym_string)
        self.batch_size = batch_size
        self.is_continuous = is_continuous
        if is_continuous is True:
            self.a_bound = [self.env.action_space.low, self.env.action_space.high]
        else:
            self.a_bound = None
