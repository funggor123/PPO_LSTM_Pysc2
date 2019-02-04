import numpy as np


class Episode:
    def __init__(self):
        self.experience = []
        self.acc_reward = 0
        self.loss = 0
        self.terminal_state_value = None

    def add_reward(self, reward):
        self.acc_reward = self.acc_reward + reward

    def add_experience(self, experience):
        self.experience.append(experience)

    def add_loss(self, loss):
        self.loss = self.loss + loss

    def set_terminal_state_value(self, terminal_state_v):
        self.terminal_state_value = terminal_state_v

    def print_average_reward(self):
        print(self.acc_reward / len(self.experience))



