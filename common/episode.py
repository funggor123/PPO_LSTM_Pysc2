import numpy as np


class Episode:
    def __init__(self):
        self.experiences = []
        self.reward = 0
        self.loss = 0
        self.terminal_state_value = None

    def add_episode(self, episode):
        self.reward = self.reward + episode.reward
        self.loss = self.loss + episode.loss

    def add_experience(self, experience):
        self.experiences.append(experience)

    def set_terminal_state_value(self, terminal_state_value):
        self.terminal_state_value = terminal_state_value

    def add_reward(self, reward):
        self.reward = self.reward + reward






