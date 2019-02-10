from algorithum.a2c import A2C
import tensorflow as tf


class PPO(A2C):

    def __init__(self, obs_dimension, a_dimension, lr, action_space_length, feature_transform,
                 epsilon, model, regular_str, minibatch, epoch):
        super(PPO, self).__init__(obs_dimension, a_dimension, action_space_length, lr,
                                  feature_transform, model, regular_str, minibatch, epoch)

        self.value_old_out, self.policy_old_out, self.old_params = model.make_network(input_opr=self.batch['state'],
                                                                                      name="old",
                                                                                      train=False)

        self.sync_network = self.get_sync_old(self.params, self.old_params)

        if self.model.is_continuous:
            entropy = self.policy_out.entropy()
            ratio = self.policy_out.prob(self.batch["actions"]) / self.policy_old_out.prob(self.batch["actions"])
        else:
            entropy = tf.reduce_sum(self.policy_out * tf.log(self.policy_out), axis=1, keepdims=True)
            ratio = self.get_discrete_prob(self.policy_out, self.batch["actions"]) / self.get_discrete_prob(
                self.policy_old_out,
                self.batch["actions"])

        surr = ratio * self.batch["advantage"]
        ratio_clip_opr = tf.clip_by_value(ratio,
                                          1 - epsilon,
                                          1 + epsilon)

        exp = tf.minimum(surr, ratio_clip_opr * self.batch["advantage"])
        entropy = tf.reduce_mean(entropy) * - self.reg_str
        self.policy_loss_opr = -tf.reduce_mean(exp) + entropy

        clipped_value = self.value_old_out + tf.clip_by_value(self.value_out - self.value_old_out, -epsilon, epsilon)
        loss_vf1 = tf.squared_difference(clipped_value, self.batch["rewards"])
        loss_vf2 = tf.squared_difference(self.value_out, self.batch["rewards"])
        self.value_loss_opr = tf.reduce_mean(tf.maximum(loss_vf1, loss_vf2)) * 0.5
        self.total_loss = self.value_loss_opr + self.policy_loss_opr

        self.min_policy_loss_opr = self.get_min(self.policy_loss_opr, self.optimizer, self.global_step)
        self.min_value_loss_opr = self.get_min_without_clip(self.value_loss_opr, self.optimizer)
        self.min_total_loss_opr = self.get_min(self.total_loss, self.optimizer,
                                               self.global_step)

        self.init_opr = tf.global_variables_initializer()
        self.saver_opr = tf.train.Saver()

    def get_sync_old(self, params, old_params):
        return [old_params.assign(params) for params, old_params in zip(params, old_params)]

    def sync_old(self, sess, feed_dict):
        sess.run([self.sync_network, self.iterator.initializer], feed_dict=feed_dict)

    def get_discrete_prob(self, policy_out, a):
        return tf.reduce_sum(policy_out * tf.one_hot(a, self.action_space_length[0], dtype=tf.float32),
                             axis=1, keep_dims=True)

    def learn(self, sess, episode):
        s, s_, a, r, v, g_adv, adv, q, experience_size = self.feature_t.transform(episode)
        feed_dict = {self.s: s,
                     self.td_error: g_adv,
                     self.a: a,
                     self.v: q
                     }
        self.sync_old(sess, feed_dict)

        while True:
            try:
                _ = sess.run(self.min_total_loss_opr)
                episode.loss = 0
            except tf.errors.OutOfRangeError:
                break
        return episode
