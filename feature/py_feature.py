import numpy as np
from pysc2.lib import actions
import common.utils as U


class PY_FeatureTransform:

    def __init__(self, ssize, msize):
        self.ssize = ssize
        self.msize = msize
        self.isize = len(actions.FUNCTIONS)

    def transform(self, rbs, sess, actor):
        max_step = len(rbs)

        valid_spatial_actions = np.zeros([max_step], dtype=np.float32)
        spatial_action_selecteds = np.zeros([max_step, self.ssize ** 2], dtype=np.float32)
        valid_non_spatial_actions = np.zeros([max_step, len(actions.FUNCTIONS)], dtype=np.float32)
        non_spatial_action_selecteds = np.zeros([max_step, len(actions.FUNCTIONS)], dtype=np.float32)
        minimaps = np.zeros([max_step, U.minimap_channel(), self.msize, self.msize], dtype=np.float32)
        screens = np.zeros([max_step, U.screen_channel(), self.ssize, self.ssize], dtype=np.float32)
        infos = np.zeros([max_step, self.isize], dtype=np.float32)

        v = np.zeros(shape=(max_step + 1, 1))
        r = np.zeros(shape=max_step)

        for t, [obs, action, next_obs, value] in enumerate(rbs):
            minimap = np.array(obs.observation['feature_minimap'], dtype=np.float32)
            minimap = np.expand_dims(U.preprocess_minimap(minimap), axis=0)
            screen = np.array(obs.observation['feature_screen'], dtype=np.float32)
            screen = np.expand_dims(U.preprocess_screen(screen), axis=0)
            info = np.zeros([1, self.isize], dtype=np.float32)
            info[0, obs.observation['available_actions']] = 1

            infos[t] = info
            screens[t] = screen
            minimaps[t] = minimap

            act_id = action.function
            act_args = action.arguments

            valid_actions = obs.observation["available_actions"]
            valid_non_spatial_actions[t, valid_actions] = 1
            non_spatial_action_selecteds[t, act_id] = 1

            args = actions.FUNCTIONS[act_id].args
            for arg, act_arg in zip(args, act_args):
                if arg.name in ('screen', 'minimap', 'screen2'):
                    ind = act_arg[1] * self.ssize + act_arg[0]
                    valid_spatial_actions[t] = 1
                    spatial_action_selecteds[t, ind] = 1

            v[t, 0] = value
            r[t] = obs.reward

        obs = rbs[-1][2]
        if obs.last():
            v[max_step] = 0
        else:
            v[max_step] = actor.get_value(sess, obs)
        return r, v, minimaps, screens, infos, spatial_action_selecteds, valid_spatial_actions, non_spatial_action_selecteds, valid_non_spatial_actions, \
               max_step
