from algorithum.a2c import A2C
import tensorflow as tf
from algorithum.py_a2c import Py_A2C


class Py_PPO(Py_A2C):

    def __init__(self, msize, ssize, lr, feature_transform, model, regular_str, minibatch, epoch, epsilon, isa2c=True,
                 training=True):
        super(Py_PPO, self).__init__(msize, ssize, lr, feature_transform, model, regular_str, minibatch, epoch, isa2c,
                                     training)
        self.epsilonx = epsilonx

        self.value_old_out, self.old_policy_out, self.old_params, _, _ = model.make_network(
            input_opr=self.input_opr_batch,
            name="old",
            num_action=self.isize,
            reuse=False)

        self.old_spatial_action_out = self.old_policy_out["spatial_action"]
        self.old_non_spatial_action_out = self.old_policy_out["non_spatial_action"]

        self.sync_network = self.get_sync_old(self.params, self.old_params)

        ratio = self.get_log(self.spatial_action_out, self.batch['spatial_action_selected'],
                             self.batch['non_spatial_action_selected']
                             , self.batch['valid_non_spatial_action'],
                             self.non_spatial_action_out,
                             self.batch['valid_spatial_action']
                             ) / self.get_log(self.old_spatial_action_out, self.batch['spatial_action_selected'],
                                              self.batch['non_spatial_action_selected']
                                              , self.batch['valid_non_spatial_action'],
                                              self.old_non_spatial_action_out, self.batch['valid_spatial_action'])

        surr = ratio * self.batch["advantage"]
        ratio_clip_opr = tf.clip_by_value(ratio,
                                          1 - self.epsilonx,
                                          1 + self.epsilonx)

        exp = tf.minimum(surr, ratio_clip_opr * self.batch["advantage"])

        self.policy_loss_opr = -tf.reduce_mean(exp)

        clipped_value = self.value_old_out + tf.clip_by_value(self.value_out - self.value_old_out, -self.epsilon,
                                                              self.epsilon)
        loss_vf1 = tf.squared_difference(clipped_value, self.batch["rewards"])
        loss_vf2 = tf.squared_difference(self.value_out, self.batch["rewards"])
        self.value_loss_opr = tf.reduce_mean(tf.maximum(loss_vf1, loss_vf2)) * 0.5
        self.total_loss = self.value_loss_opr + self.policy_loss_opr

        self.min_policy_loss_opr = self.get_min(self.policy_loss_opr, self.optimizer, self.global_step)
        self.min_value_loss_opr = self.get_min_without_clip(self.value_loss_opr, self.optimizer)
        self.min_total_loss_opr = self.get_min(self.total_loss, self.optimizer,
                                               self.global_step)

        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

    def get_sync_old(self, params, old_params):
        return [old_params.assign(params) for params, old_params in zip(params, old_params)]

    def get_init_opr(self):
        return self.init

    def get_saver_opr(self):
        return self.saver

    def sync_old(self, sess, feed_dict):
        sess.run([self.sync_network, self.iterator.initializer], feed_dict=feed_dict)

    def learn(self, sess, rbs, lr):
        r, v, g_adv, adv, q, minimaps, screens, infos, spatial_action_selecteds, valid_spatial_actions, non_spatial_action_selecteds, valid_non_spatial_actions, \
        max_step = self.feature_t.transform(rbs, sess, self)

        feed_dict = {self.minimap: minimaps,
                     self.screen: screens,
                     self.info: infos,
                     self.td_error: g_adv,
                     self.valid_spatial_action: valid_spatial_actions,
                     self.spatial_action_selected: spatial_action_selecteds,
                     self.valid_non_spatial_action: valid_non_spatial_actions,
                     self.non_spatial_action_selected: non_spatial_action_selecteds,
                     self.v: q,
                     self.learning_rate: lr
                     }
        self.sync_old(sess, feed_dict)

        while True:
            try:
                _, loss, global_step = self.update(sess, self.min_total_loss, self.total_loss,
                                                   self.global_step, feed_dict)
            except tf.errors.OutOfRangeError:
                break
        return loss


