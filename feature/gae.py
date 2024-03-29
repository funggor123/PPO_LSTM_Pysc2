import numpy as np
from feature.feature import FeatureTransform
from feature.runnning_stat import RunningStats

class GAE(FeatureTransform):

    def __init__(self, obs_dimension, a_dimension, gamma, beta, is_continuous, max_reward, min_reward):
        super(GAE, self).__init__(obs_dimension, a_dimension, is_continuous, max_reward, min_reward)
        self.gamma = gamma
        self.beta = beta

    def transform(self, episode):
        s, s_, a, r, v, experience_size = super(GAE, self).transform(episode)

        adv = np.zeros(shape=experience_size)
        g_adv = np.zeros(shape=(experience_size, 1))
        q = np.zeros(shape=(experience_size, 1))

        for t in range(experience_size):
            q[t, 0] = r[t] + self.gamma * v[t + 1, 0]
            adv[t] = q[t, 0] - v[t, 0]

        for t in range(experience_size):
            for t_ in range(t, experience_size):
                g_adv[t, 0] = g_adv[t, 0] + (self.gamma * self.beta) ** (t_ - t) * adv[t_]

        for t in range(experience_size):
            q[t, 0] = g_adv[t, 0] + v[t, 0]
        '''
        g_adv = (g_adv - g_adv.mean()) / np.maximum(g_adv.std(), 1e-6)
        r = (r - self.min_reward / 2) / ((self.max_reward - self.min_reward)/2)
        '''
        return s, s_, a, r, v, g_adv, adv, q, experience_size
