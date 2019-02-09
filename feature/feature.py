import numpy as np
from common.experience import Experience


class FeatureTransform:

    def __init__(self, obs_dimension, a_dimension, is_continuous):
        self.obs_dimension = obs_dimension
        self.a_dimension = a_dimension
        self.is_continuous = is_continuous

    def transform(self, episode):
        experiences = episode.experiences
        max_step = len(experiences)

        s = np.zeros(shape=(max_step,) + self.obs_dimension)
        s_ = np.zeros(shape=(max_step,) + self.obs_dimension)
        v = np.zeros(shape=(max_step + 1, 1)
                     )
        if self.is_continuous:
            a = np.zeros(shape=(max_step,) + self.a_dimension)
        else:
            a = np.zeros(shape=max_step)

        r = np.zeros(shape=max_step)

        for t, exp in enumerate(experiences):
            s[t] = np.reshape(exp.last_state_obs, newshape=(1,) + self.obs_dimension)
            s_[t] = np.reshape(exp.current_state_obs, newshape=(1,) + self.obs_dimension)
            v[t, 0] = exp.last_state_value
            r[t] = exp.reward
            a[t] = exp.action

        v[max_step] = episode.terminal_state_value
        return s, s_, a, r, v, max_step
