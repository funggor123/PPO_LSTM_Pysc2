import tensorflow as tf
from tensorflow import layers
import numpy as np


class A2C:

    def __init__(self, observe_space_len, action_space_len, learning_rate, feature_transform, model):

        self.s_len = observe_space_len
        self.a_len = action_space_len
        self.lr = learning_rate
        self.model = model
        self.feature_transform = feature_transform

        self.s = tf.placeholder(tf.float64, shape=(None,) + observe_space_len, name="state")
        self.v = tf.placeholder(tf.float64, shape=None, name="value")
        self.td_error = tf.placeholder(tf.float64, shape=None, name="td_error")

        if model.is_continuous:
            self.a = tf.placeholder(tf.float64, shape=(None, action_space_len), name="action")
        else:
            self.a = tf.placeholder(tf.float64, shape=None, name="action")

        self.value_opr, self.policy_opr, self.params, self.t = model.make_network(input_opr=self.s,
                                                                                  name="target",
                                                                                  train=True)

        self.value_loss_opr = self.__get_value_loss_opr__(self.value_opr, self.v)

        self.optimizer_opr = self.__get_optimizer_opr__(self.lr)

        if model.is_continuous:
            self.policy_loss_opr = self.__get_policy_continous_loss_opr__(self.policy_opr, self.a, self.td_error)
            self.out_opr = tf.squeeze(self.policy_opr.sample(1), axis=0)
        else:
            self.policy_loss_opr = self.__get_policy_discrete_loss_opr__(self.policy_opr, self.a, self.td_error)
            self.out_opr = tf.nn.softmax(self.policy_opr)

        self.total_loss_opr = self.__get_total_loss_opr__(self.value_loss_opr, self.policy_loss_opr)

        '''
        with tf.name_scope('Loss'):
            tf.summary.scalar('loss', self.total_loss_opr)
        '''

        self.global_step = tf.train.create_global_step()
        '''
        self.min_policy_loss_opr = self.__get_min_opr__(self.policy_loss_opr, self.optimizer_opr, self.global_step)
        self.min_value_loss_opr = self.__get_min_opr_without_gc__(self.value_loss_opr, self.optimizer_opr)
        '''

        self.min_policy_loss_opr = self.__get_min_with_gc_opr__(self.policy_loss_opr, self.optimizer_opr,
                                                                self.global_step)

        self.min_value_loss_opr = self.__get_min_with_gc_opr_without_gs__(self.value_loss_opr, self.optimizer_opr)

        self.min_total_loss_opr = self.__get_min_with_gc_opr__(self.total_loss_opr, self.optimizer_opr,
                                                               self.global_step)

        '''
        self.summary_opr = tf.summary.merge_all()
        '''

        self.init_opr = tf.global_variables_initializer()
        self.saver_opr = tf.train.Saver()

        '''
        self.writer = tf.summary.FileWriter("TensorBoard/", graph=graph)
        self.graph = graph
        '''

    def get_global_step_var(self):
        return self.global_step

    def get_summary_opr(self):
        return self.summary_opr

    def get_init_opr(self):
        return self.init_opr

    def get_saver_opr(self):
        return self.saver_opr

    def __get_min_opr__(self, loss, opt_opr, global_step):
        return opt_opr.minimize(loss, global_step=global_step)

    def __get_min_opr_without_gc__(self, loss, opt_opr):
        return opt_opr.minimize(loss)

    ## Operators and Variables

    def ClipIfNotNone(self, grad):
        if grad is None:
            return grad
        return tf.clip_by_value(grad, -1, 1)

    def __get_min_with_gc_opr__(self, loss, opt_opr, global_step):
        gvs = opt_opr.compute_gradients(loss)

        clipped_gradients = [(self.ClipIfNotNone(grad), var) for grad, var in gvs]
        return opt_opr.apply_gradients(clipped_gradients, global_step=global_step)

    def __get_min_with_gc_opr_without_gs__(self, loss, opt_opr):
        gvs = opt_opr.compute_gradients(loss)
        clipped_gradients = [(self.ClipIfNotNone(grad), var) for grad, var in gvs]
        return opt_opr.apply_gradients(clipped_gradients)

    def __get_value_loss_opr__(self, value_out, v):
        return tf.reduce_mean(tf.square((value_out - v)))

    def __get_policy_discrete_loss_opr__(self, policy_out, a, td_error):
        return -tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=policy_out, labels=a) * td_error)

    def __get_policy_continous_loss_opr__(self, policy_out, a, td_error):
        log_pi = tf.log(tf.clip_by_value(policy_out.prob(a), 1e-20, 1.0))
        return -tf.reduce_mean(tf.log(tf.reduce_sum(log_pi, 1)) * td_error)

    def __get_total_loss_opr__(self, value_loss, policy_loss):
        return tf.add(value_loss, policy_loss)

    def __get_optimizer_opr__(self, lr):
        return tf.train.AdamOptimizer(lr)

    ## Session Run

    def __update_network__(self, sess, min_opr, loss_opr, global_step, feed_dict):
        return sess.run([min_opr, loss_opr, global_step], feed_dict)

    def __update_network_without_gc__(self, sess, min_opr, loss_opr, feed_dict):
        return sess.run([min_opr, loss_opr], feed_dict)

    def __get_value__(self, sess, s, value_opr, s_placeholder):
        return sess.run(value_opr, feed_dict={s_placeholder: s})

    def __get_td_error__(self, sess, s, q, value_opr, s_placeholder):
        return q - self.__get_value__(sess, s, value_opr, s_placeholder)

    def __get_compute_gradient__(self, loss, opt):
        return opt.compute_gradients(loss)

    def get_value(self, sess, s):
        s = np.reshape(s, newshape=(1,) + self.s_len)
        value = sess.run(self.value_opr, feed_dict={self.s: s})
        return value

    ## Learn

    def learn(self, sess, episode):
        s, s_, a, r, v, g_adv, adv, q, experience_size = self.feature_transform.transform(episode)
        feed_dict = {self.s: s,
                     self.td_error: g_adv,
                     self.a: a,
                     self.v: q
                     }

        _, loss, global_step = self.__update_network__(sess, self.min_total_loss_opr, self.total_loss_opr,
                                                       self.global_step,

                                                       feed_dict)
        episode.loss = loss

        return episode, global_step

    ## Choose Action

    def choose_action_with_exploration(self, sess, s):
        shape = (1,) + self.s_len
        s = np.reshape(s, newshape=shape)
        action, value = sess.run([self.out_opr, self.value_opr], feed_dict={self.s: s})
        if self.model.is_continuous:
            action = np.reshape(action, newshape=self.a_len)
            return [np.clip(action[0], -1, 1), np.clip(action[1], 0, 1), np.clip(action[2], 0, 1)], value
        else:
            return np.random.choice(np.arange(action.shape[1]), p=action.ravel()), value

    def choose_action(self, sess, s):
        shape = (1,) + self.s_len
        s = np.reshape(s, newshape=shape)
        action, value = sess.run([self.out_opr, self.value_opr], feed_dict={self.s: s})
        if self.model.is_continuous:
            action = np.reshape(action, newshape=self.a_len)
            return action, value
        else:
            return np.argmax(action.ravel()), value
