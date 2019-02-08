from a2c import A2C
import tensorflow as tf


class PPO(A2C):

    def __init__(self, observe_space_len, action_space_dim, learning_rate, action_space_length, feature_transform,
                 clip_r, model, regularization_stength):
        super(PPO, self).__init__(observe_space_len, action_space_dim, action_space_length, learning_rate,
                                  feature_transform, model, regularization_stength)

        self.policy_behave_opr, self.behave_params = model.make_actor_network(input_opr=self.s,
                                                                              name="behave",
                                                                              train=False)

        self.value_behave_opr, _ = model.make_critic_network(input_opr=self.s,
                                                             name="behave_value",
                                                             train=False)

        self.sync_behave_opr = self.__get_sync_behave_opr__(self.params, self.behave_params)
        with tf.variable_scope('surr'):
            if model.is_continuous:
                entropy = self.policy_opr.entropy()
                ratio_opr = self.policy_opr.prob(self.a) / self.policy_behave_opr.prob(self.a)
            else:
                entropy = tf.reduce_sum(self.policy_opr[0] * tf.log(self.policy_opr[0]), axis=1)
                ratio_opr = tf.reduce_sum(
                    self.policy_opr[0] * tf.one_hot(self.a[0], self.action_space_length[0], dtype=tf.float64), axis=1) / \
                            tf.reduce_sum(self.policy_behave_opr[0] * tf.one_hot(self.a[0], self.action_space_length[0],
                                                                                 dtype=tf.float64), axis=1)
                for i in range(1, len(self.a_dim)):
                    ratio_opr = tf.reduce_sum(
                        self.policy_opr[i] * tf.one_hot(self.a[i], self.action_space_length[i], action_space_dim[i],
                                                        dtype=tf.float64), axis=1) / \
                                tf.reduce_sum(
                                    self.policy_behave_opr[i] * tf.one_hot(self.a[i], self.action_space_length[i],
                                                                           dtype=tf.float64), axis=1) + ratio_opr
            surr = ratio_opr * self.td_error
            ratio_clip_opr = tf.clip_by_value(ratio_opr,
                                              1 - clip_r,
                                              1 + clip_r)

            exp = tf.minimum(surr, ratio_clip_opr * self.td_error)
            entropy = -entropy * self.reg_s
            self.policy_loss_opr = -tf.reduce_mean(exp) + tf.reduce_mean(entropy)

            '''
                   clipped_value = self.value_behave_opr + tf.clip_by_value(self.v - self.value_behave_opr, -clip_r, clip_r)
                   loss_vf1 = tf.squared_difference(clipped_value, self.value_opr)
                   loss_vf2 = tf.squared_difference(self.v, self.value_opr)
                   self.value_loss_opr = tf.reduce_mean(tf.maximum(loss_vf1, loss_vf2)) * 0.5
                   '''

    def __get_sync_behave_opr__(self, params, behave_params):
        return [behave_params.assign(params) for params, behave_params in zip(params, behave_params)]

    def sync_target(self, sess):
        sess.run(self.sync_behave_opr)

    ## Learn

    def learn(self, sess, episode):
        self.sync_target(sess)
        s, s_, a, r, v, g_adv, adv, q, experience_size = self.feature_transform.transform(episode)

        feed_dict = {self.s: s,
                     self.td_error: g_adv,
                     self.a: a,
                     self.v: q
                     }
        global_step = 0

        for _ in range(10):
            _, loss, global_step = self.__update_network__(sess, self.min_policy_loss_opr,
                                                           self.policy_loss_opr,
                                                           self.global_step,
                                                           feed_dict)
            episode.loss = loss
            _, loss, global_step = self.__update_network__(sess, self.min_value_loss_opr,
                                                           self.value_loss_opr,
                                                           self.global_step,
                                                           feed_dict)

        return episode, global_step
