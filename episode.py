import numpy as np


class Episode:
    def __init__(self):
        self.experience = []
        self.reward = 0
        self.step = 0

    def add_reward(self, reward):
        self.reward = self.reward + reward

    def add_experience(self, experience):
        self.experience.append(experience)

    def add_step(self):
        self.step = self.step + 1

    def print_average_reward(self):
        print(self.reward / self.step)

