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

        self.value_behave_opr, self.value_behave_params = model.make_critic_network(input_opr=self.s,
                                                                                    name="behave_value",
                                                                                    train=False)

        self.sync_behave_opr = self.__get_sync_behave_opr__(self.params, self.behave_params)
        self.sync_value_behave_opr = self.__get_sync_behave_opr__(self.value_params, self.value_behave_params)

        if model.is_continuous:
            entropy = self.policy_opr.entropy()
            ratio_opr = self.policy_opr.prob(self.a) / self.policy_behave_opr.prob(self.a)
        else:
            entropy = tf.reduce_sum(self.policy_opr * tf.log(self.policy_opr), axis=1, keep_dims=True)
            ratio_opr = self.discrete_prob(self.policy_opr, self.a) / self.discrete_prob(self.policy_behave_opr, self.a)

        surr1 = self.td_error * ratio_opr
        surr2 = self.td_error * tf.clip_by_value(ratio_opr,
                                                 1 - clip_r,
                                                 1 + clip_r)

        loss_pi = -tf.reduce_mean(tf.minimum(surr1, surr2))
        entropy = -self.reg_s * tf.reduce_mean(entropy)
        self.policy_loss_opr = loss_pi + entropy

        '''
        clipped_value = self.value_behave_opr + tf.clip_by_value(self.v - self.value_behave_opr, -clip_r, clip_r)
        loss_vf1 = tf.squared_difference(clipped_value, self.value_opr)
        loss_vf2 = tf.squared_difference(self.v, self.value_opr)
        self.value_loss_opr = tf.reduce_mean(tf.maximum(loss_vf1, loss_vf2)) * 0.5
        '''

        self.min_policy_loss_opr = self.__get_min_opr__(self.policy_loss_opr, self.optimizer_opr, self.global_step)
        self.min_value_loss_opr = self.__get_min_opr_without_gc__(self.value_loss_opr, self.optimizer_opr)
        self.min_total_loss_opr = self.__get_min_opr__(self.total_loss_opr, self.optimizer_opr,
                                                       self.global_step)

    def __get_sync_behave_opr__(self, params, behave_params):
        return [behave_params.assign(params) for params, behave_params in zip(params, behave_params)]

    def sync_target(self, sess):
        sess.run(self.sync_behave_opr)
        sess.run(self.sync_value_behave_opr)

    def discrete_prob(self, policy_out, a):
        return tf.reduce_sum(policy_out * tf.one_hot(a, self.action_space_length[0], dtype=tf.float32),
                             axis=1, keep_dims=True)

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
