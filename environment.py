import gym
from gym import wrappers

class Environment:

    def __init__(self, action_space_length, observation_space_length, gym_string, max_step, batch_size):
        self.action_space_length = action_space_length
        self.observation_space_length = observation_space_length
        self.gym_string = gym_string
        self.max_step = max_step
        self.env = gym.make(self.gym_string)
        self.env = wrappers.Monitor(self.env, "/tmp", video_callable=False, force=True)
        self.batch_size = batch_size





