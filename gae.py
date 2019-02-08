import numpy as np
from feature import FeatureTransform


class GAE(FeatureTransform):
    """
    state_len : dimension of state
    action_len : dimension of action
    discount_rate : 0 < x < 1
    n_step_rate : 0 < x < 1
    """

    def __init__(self, state_len, action_len, action_dim, discount_rate, n_step_rate):
        super(GAE, self).__init__(state_len, action_len, action_dim)
        self.discount_rate = discount_rate
        self.n_step_rate = n_step_rate

    def transform(self, episode):
        s, s_, a, r, v, experience_size = super(GAE, self).transform(episode)

        adv = np.zeros(shape=experience_size)
        g_adv = np.zeros(shape=(experience_size, 1))
        q = np.zeros(shape=(experience_size, 1))

        for t in range(experience_size):
            q[t, 0] = r[t] + self.discount_rate * v[t + 1, 0]
            adv[t] = q[t, 0] - v[t, 0]

        for t in range(experience_size):
            for t_ in range(t, experience_size):
                g_adv[t, 0] = g_adv[t, 0] + (self.discount_rate * self.n_step_rate) ** (t_-t) * adv[t_]

        for t in range(experience_size):
            q[t, 0] = g_adv[t, 0] + v[t, 0]

        return s, s_, a, r, v, g_adv, adv, q, experience_size
