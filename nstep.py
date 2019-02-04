import numpy as np
from feature import FeatureTransform
'''

class GAE(FeatureTransform):
    """
    state_len : dimension of state
    action_len : dimension of action
    discount_rate : 0 < x < 1
    n_step_rate : 0 < x < 1
    """

    def __init__(self, state_len, action_len, discount_rate, n_step_rate):
        super(GAE, self).__init__(state_len, action_len)
        self.discount_rate = discount_rate
        self.n_step_rate = n_step_rate

    def transform(self, episode):
        for step, exp in enumerate(experience):
            if step + self.n_step < experience_size:
                q[step] = experience[step + self.n_step].last_state_value
                for i in reversed(range(step, step + self.n_step)):
                    q[step] = experience[i].reward + self.discount_rate * q[step]
                    q_[step] = experience[i].reward + self.discount_rate * self.discount_rate * q[step]
            else:
                q[step] = episode.terminal_state_value
                for i in reversed(range(step, experience_size)):
                    q[step] = experience[i].reward + self.discount_rate * q[step]
                    q_[step] = experience[i].reward + self.discount_rate * self.discount_rate * q[step]

            s[step] = exp.last_state
            a[step] = exp.action

        return s, s_, a, r, v, gadv, experience_size

'''
