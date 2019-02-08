import numpy as np


class FeatureTransform:

    def __init__(self, state_len, action_len, action_dim):
        self.state_len = state_len
        self.action_len = action_len
        self.action_dim = action_dim

    def transform(self, episode):
        experience = episode.experience
        max_t = len(experience)

        s = np.zeros(shape=(max_t,) + self.state_len)
        s_ = np.zeros(shape=(max_t,) + self.state_len)
        v = np.zeros(shape=(max_t+1, 1))
        a = np.zeros(shape=(max_t,) + self.action_dim)
        r = np.zeros(shape=max_t)

        for t, exp in enumerate(experience):
            s[t] = np.reshape(exp.last_state, newshape=(1,) + self.state_len)
            s_[t] = np.reshape(exp.current_state, newshape=(1,) + self.state_len)
            v[t, 0] = exp.last_state_value
            r[t] = exp.reward
            a[t] = exp.action

        v[max_t] = episode.terminal_state_value
        return s, s_, a, r, v, max_t
