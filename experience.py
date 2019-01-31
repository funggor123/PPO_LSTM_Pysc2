import numpy as np


class Experience:
    def __init__(self, obs_len, act_len):
        self.reward = None
        self.action = None
        self.last_state = None
        self.current_state = None
        self.last_state_value = None
        self.grads = None

        self.obs_len = obs_len
        self.act_len = act_len

    def set_reward(self, reward):
        self.reward = reward

    def set_action(self, action):
        self.action = action

    def set_last_state(self, last_state):
        self.last_state = last_state

    def set_current_state(self, current_state):
        self.current_state = current_state

    def set_last_state_value(self, last_state_value):
        self.last_state_value = last_state_value
    

