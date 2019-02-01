import tensorflow as tf
from tensorflow import layers
import numpy as np
from feature import feature_transform


class A2C:

    def __init__(self, observe_space_len, action_space_len, discount_ratio, num_units, num_layers, activation,
                 learning_rate, n_step):

        self.s = tf.placeholder(tf.float64, shape=(None, observe_space_len), name="state")
        self.v = tf.placeholder(tf.float64, shape=None, name="value")
        self.td_error = tf.placeholder(tf.float64, shape=None, name="td_error")
        self.a = tf.placeholder(tf.int32, shape=None, name="action")

        self.s_len = observe_space_len
        self.a_len = action_space_len
        self.d_r = discount_ratio
        self.lr = learning_rate

        self.value_opr, self.policy_opr, self.params = self.get_network_opr(self.s, num_units, num_layers, activation,
                                                                            "target")

        self.value_loss_opr = self.__get_value_loss_opr__(self.value_opr, self.v)
        self.policy_loss_opr = self.__get_policy_loss_opr__(self.policy_opr, self.a, self.td_error)

        self.optimizer_opr = self.__get_optimizer_opr__(self.lr)

        self.softmax_policy_opr = tf.nn.softmax(self.policy_opr)

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

        self.min_policy_loss_opr = self.__get_min_opr__(self.policy_loss_opr, self.optimizer_opr,
                                                                self.global_step)

        self.min_value_loss_opr = self.__get_min_opr_without_gc__(self.value_loss_opr, self.optimizer_opr)

        self.min_total_loss_opr = self.__get_min_with_gc_opr__(self.total_loss_opr, self.optimizer_opr,
                                                               self.global_step)

        '''
        self.summary_opr = tf.summary.merge_all()
        '''

        self.init_opr = tf.global_variables_initializer()
        self.saver_opr = tf.train.Saver()

        self.n_step = n_step

        '''
        self.writer = tf.summary.FileWriter("TensorBoard/", graph=graph)
        self.graph = graph
        '''

    def get_network_opr(self, input_opr, num_units, num_layers, activation, name, train=True):

        with tf.variable_scope(name):
            init_xavier = tf.contrib.layers.xavier_initializer()
            dense_out = layers.dense(input_opr, units=num_units, activation=activation, kernel_initializer=init_xavier,
                                     trainable=train)
            for n in range(0, num_layers):
                dense_out = layers.dense(dense_out, units=num_units, activation=activation,
                                         kernel_initializer=init_xavier, trainable=train)
            policy_out = layers.dense(dense_out, units=self.a_len, activation=activation,
                                      kernel_initializer=init_xavier, trainable=train)
            value_out = layers.dense(dense_out, units=1, activation=activation, kernel_initializer=init_xavier,
                                     trainable=train)

        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)

        return value_out, policy_out, params

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

    def __get_min_with_gc_opr__(self, loss, opt_opr, global_step):
        gvs = opt_opr.compute_gradients(loss)

        def ClipIfNotNone(grad):
            if grad is None:
                return grad
            return tf.clip_by_value(grad, -1, 1)

        clipped_gradients = [(ClipIfNotNone(grad), var) for grad, var in gvs]
        return opt_opr.apply_gradients(clipped_gradients, global_step=global_step)

    def __get_min_with_gc_opr_without_gs__(self, loss, opt_opr):
        gvs = opt_opr.compute_gradients(loss)

        def ClipIfNotNone(grad):
            if grad is None:
                return grad
            return tf.clip_by_value(grad, -1, 1)

        clipped_gradients = [(ClipIfNotNone(grad), var) for grad, var in gvs]
        return opt_opr.apply_gradients(clipped_gradients)

    def __get_value_loss_opr__(self, value_out, v):
        return tf.reduce_mean(tf.square((value_out - v)))

    def __get_policy_loss_opr__(self, policy_out, a, td_error):
        return -tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=policy_out, labels=a) * td_error)

    def __get_total_loss_opr__(self, value_loss, policy_loss):
        return value_loss + policy_loss

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
        s = np.reshape(s, newshape=(1, self.s_len))
        value = sess.run(self.value_opr, feed_dict={self.s: s})
        return value

    ## Learn

    def learn(self, sess, episode):
        s, a, q, q_ = feature_transform(self, episode)
        td_error = self.__get_td_error__(sess, s, q, self.value_opr, self.s)

        feed_dict = {self.s: s,
                     self.td_error: td_error,
                     self.a: a,
                     self.v: q_
                     }

        _, loss, global_step = self.__update_network__(sess, self.min_total_loss_opr, self.total_loss_opr,
                                                       self.global_step,
                                                       feed_dict)

        episode.loss = loss
        '''
        _, loss, global_step = self.__update_network__(sess, self.min_policy_loss_opr, self.policy_loss_opr,
                                                       self.global_step,
                                                       feed_dict)

        episode.loss = loss

        feed_dict = {self.s: s, self.v: q}
        _, loss = self.__update_network_without_gc__(sess, self.min_value_loss_opr, self.value_loss_opr,
                                                     feed_dict)
        '''
        return episode, global_step

    ## Choose Action

    def choose_action_with_exploration(self, sess, s):
        s = np.reshape(s, newshape=(1, self.s_len))
        action, value = sess.run([self.softmax_policy_opr, self.value_opr], feed_dict={self.s: s})
        return np.random.choice(np.arange(action.shape[1]), p=action.ravel()), value

    def choose_action(self, sess, s):
        s = np.reshape(s, newshape=(1, self.s_len))
        action, value = sess.run([self.softmax_policy_opr, self.value_opr], feed_dict={self.s: s})
        return np.argmax(action.ravel()), value
