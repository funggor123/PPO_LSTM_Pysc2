from algorithum.a2c import A2C
import tensorflow as tf


class PPO(A2C):

    def __init__(self, obs_dimension, a_dimension, lr, action_space_length, feature_transform,
                 epsilon, model, regular_str):
        super(PPO, self).__init__(obs_dimension, a_dimension, action_space_length, lr,
                                  feature_transform, model, regular_str)

        self.policy_old_out, self.policy_old_params = model.make_actor_network(input_opr=self.s,
                                                                               name="old_policy",
                                                                               train=False)

        self.value_old_out, self.value_old_params = model.make_critic_network(input_opr=self.s,
                                                                              name="old_value",
                                                                              train=False)

        self.sync_old_policy = self.get_sync_old(self.policy_params, self.policy_old_params)
        self.sync_old_value = self.get_sync_old(self.value_params, self.value_old_params)

        if self.model.is_continuous:
            entropy = self.policy_out.entropy()
            ratio = self.policy_out.prob(self.a) / self.policy_old_out.prob(self.a)
        else:
            entropy = tf.reduce_sum(self.policy_out * tf.log(self.policy_out), axis=1, keep_dims=True)
            ratio = self.get_discrete_prob(self.policy_out, self.a) / self.get_discrete_prob(self.policy_old_out,
                                                                                             self.a)

        surr1 = self.td_error * ratio
        surr2 = self.td_error * tf.clip_by_value(ratio,
                                                 1 - epsilon,
                                                 1 + epsilon)

        loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
        entropy = -self.reg_str * tf.reduce_mean(entropy)
        self.policy_loss = loss + entropy

        clipped_value = self.value_old_out + tf.clip_by_value(self.v - self.value_old_out, -epsilon, epsilon)
        loss_vf1 = tf.squared_difference(clipped_value, self.value_out)
        loss_vf2 = tf.squared_difference(self.v, self.value_out)
        self.value_loss = tf.reduce_mean(tf.maximum(loss_vf1, loss_vf2)) * 0.5

    def get_sync_old(self, params, old_params):
        return [old_params.assign(params) for params, old_params in zip(params, old_params)]

    def sync_old(self, sess):
        sess.run(self.sync_old_policy)
        sess.run(self.sync_old_value)

    def get_discrete_prob(self, policy_out, a):
        return tf.reduce_sum(policy_out * tf.one_hot(a, self.action_space_length[0], dtype=tf.float32),
                             axis=1, keep_dims=True)

    def learn(self, sess, episode):
        self.sync_old(sess)
        s, s_, a, r, v, g_adv, adv, q, experience_size = self.feature_t.transform(episode)

        feed_dict = {self.s: s,
                     self.td_error: g_adv,
                     self.a: a,
                     self.v: q
                     }

        for _ in range(10):
            _, loss, global_step = self.update(sess, self.min_policy_loss,
                                               self.policy_loss,
                                               self.global_step,
                                               feed_dict)
            episode.loss = loss
            _, loss, global_step = self.update(sess, self.min_value_loss,
                                               self.value_loss,
                                               self.global_step,
                                               feed_dict)

        return episode
