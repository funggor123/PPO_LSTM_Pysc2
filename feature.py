import numpy as np


class FeatureTransform:

    def __init__(self, state_len, action_len):
        self.state_len = state_len
        self.action_len = action_len

    def transform(self, episode):
        experience = episode.experience
        max_t = len(experience)

        s = np.zeros(shape=(max_t,)+self.state_len)
        s_ = np.zeros(shape=(max_t,)+self.state_len)
        v = np.zeros(dtype=np.float64, shape=(max_t+1))

        a = np.zeros(shape=(max_t, self.action_len))
        r = np.zeros(shape=max_t)

        for t, exp in enumerate(experience):
            s[t] = exp.last_state
            s_[t] = exp.current_state
            v[t] = exp.last_state_value
            r[t] = exp.reward
            a[t] = exp.action

        v[max_t] = episode.terminal_state_value
        return s, s_, a, r, v, max_t
