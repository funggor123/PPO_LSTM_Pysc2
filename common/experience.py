import numpy as np


class Experience:
    def __init__(self):
        self.reward = None
        self.action = None
        self.last_state_obs = None
        self.current_state_obs = None
        self.last_state_value = None

    def set_all(self, reward, action, last_state_obs, current_state_obs, last_state_value):
        self.last_state_value = last_state_value
        self.current_state_obs = current_state_obs
        self.last_state_obs = last_state_obs
        self.action = action
        self.reward = reward

    

