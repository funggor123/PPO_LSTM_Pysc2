import numpy as np


class Episode:
    def __init__(self):
        self.experience = []
        self.reward = 0
        self.loss = 0
        self.terminal_state_value = None

    def add_reward(self, step_reward):
        self.reward = self.reward + step_reward

    def add_experience(self, experience):
        self.experience.append(experience)

    def set_loss(self, loss):
        self.loss = loss

    def set_terminal_state_value(self, terminal_state_v):
        self.terminal_state_value = terminal_state_v



