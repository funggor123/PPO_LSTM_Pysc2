import numpy as np
from feature.py_feature import PY_FeatureTransform


class PY_GAE(PY_FeatureTransform):

    def __init__(self, ssize, msize, gamma, beta):
        super(PY_GAE, self).__init__(ssize, msize)
        self.gamma = gamma
        self.beta = beta

    def transform(self, rbs,  sess, actor):
        r, v, minimaps, screens, infos, spatial_action_selecteds, valid_spatial_actions, non_spatial_action_selecteds, valid_non_spatial_actions, \
        max_step = super(PY_GAE, self).transform(rbs, sess, actor)

        adv = np.zeros(shape=max_step)
        g_adv = np.zeros(shape=(max_step, 1))
        q = np.zeros(shape=(max_step, 1))

        for t in range(max_step):
            q[t, 0] = r[t] + self.gamma * v[t + 1, 0]
            adv[t] = q[t, 0] - v[t, 0]

        for t in range(max_step):
            for t_ in range(t, max_step):
                g_adv[t, 0] = g_adv[t, 0] + (self.gamma * self.beta) ** (t_ - t) * adv[t_]

        for t in range(max_step):
            q[t, 0] = g_adv[t, 0] + v[t, 0]

        return r, v, g_adv, adv, q, minimaps, screens, infos, spatial_action_selecteds, valid_spatial_actions, non_spatial_action_selecteds, valid_non_spatial_actions, \
        max_step
