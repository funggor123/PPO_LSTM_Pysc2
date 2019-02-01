import numpy as np


def feature_transform(actor, episode):
    experience = episode.experience
    experience_size = len(experience)

    s = np.zeros(shape=(experience_size, actor.s_len))
    q = np.zeros(dtype=np.float64, shape=(experience_size, 1))
    q_ = np.zeros(dtype=np.float64, shape=(experience_size, 1))
    a = np.zeros(shape=experience_size)

    for step, exp in enumerate(experience):
        if step + actor.n_step < experience_size:
            q[step] = experience[step + actor.n_step].last_state_value
            for i in reversed(range(step, step + actor.n_step)):
                q[step] = experience[i].reward + actor.d_r * q[step]
                q_[step] = experience[i].reward + actor.d_r * actor.d_r * q[step]
        else:
            q[step] = episode.terminal_state_value
            for i in reversed(range(step, experience_size)):
                q[step] = experience[i].reward + actor.d_r * q[step]
                q_[step] = experience[i].reward + actor.d_r * actor.d_r * q[step]

        s[step] = exp.last_state
        a[step] = exp.action
    return s, a, q, q_
