from a2c import A2C
import tensorflow as tf
import numpy as np
from feature import feature_transform


class PPO(A2C):

    def __init__(self, observe_space_len, action_space_len, discount_ratio, num_units, num_layers, activation,
                 learning_rate, n_step, clip_r):
        super(PPO, self).__init__(observe_space_len, action_space_len, discount_ratio, num_units, num_layers,
                                  activation, learning_rate, n_step)

        self.clip_min = 1.0 - clip_r
        self.clip_max = 1.0 + clip_r

        self.s_b = tf.placeholder(tf.float64, shape=(None, observe_space_len), name="state")

        _, self.policy_behave_opr, self.behave_params = self.get_network_opr(self.s_b, num_units,
                                                                             num_layers,
                                                                             activation,
                                                                             "behave",
                                                                             train=False)

        self.sync_behave_opr = self.__get_sync_behave_opr__(self.params, self.behave_params)

        self.softmax_policy_behave_opr = tf.nn.softmax(self.policy_behave_opr)

        ratio_opr = (tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.policy_opr, labels=self.a)
                     / tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.policy_behave_opr,
                                                                      labels=self.a))
        surr = ratio_opr * self.td_error
        ratio_clip_opr = tf.clip_by_value(ratio_opr,
                                          self.clip_min,
                                          self.clip_max)

        self.policy_loss_opr = -tf.reduce_mean(tf.minimum(surr, ratio_clip_opr * self.td_error))

    def __get_sync_behave_opr__(self, params, behave_params):
        return [behave_params.assign(params) for params, behave_params in zip(params, behave_params)]

    def sync_target(self, sess):
        sess.run(self.sync_behave_opr)

    ## Learn

    def learn(self, sess, episode):
        s, a, q, q_ = feature_transform(self, episode)
        td_error = self.__get_td_error__(sess, s, q, self.value_opr, self.s)

        feed_dict = {self.s: s,
                     self.td_error: td_error,
                     self.a: a,
                     self.s_b: s
                     }
        global_step = 0
        for _ in range(10):
            _, loss, global_step = self.__update_network__(sess, self.min_policy_loss_opr,
                                                           self.policy_loss_opr,
                                                           self.global_step,
                                                           feed_dict)
            episode.loss = loss
        for _ in range(10):
            feed_dict = {self.s: s,
                         self.v: q_}
            _, loss = self.__update_network_without_gc__(sess, self.min_value_loss_opr, self.value_loss_opr,
                                                         feed_dict)

        return episode, global_step
