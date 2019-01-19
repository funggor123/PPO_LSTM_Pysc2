import tensorflow as tf
from tensorflow import layers
import numpy as np


class A2C:

    def __init__(self, observe_space_len, action_space_len, discount_ratio, num_units, num_layers, activation,
                 learning_rate, is_training=True):

        self.s = tf.placeholder(tf.float64, shape=(None, observe_space_len), name="state")
        self.v = tf.placeholder(tf.float64, shape=None, name="value")
        self.td_error = tf.placeholder(tf.float64, shape=None, name="td_error")
        self.a = tf.placeholder(tf.int32, shape=None, name="action")

        self.s_len = observe_space_len
        self.a_len = action_space_len
        self.d_r = discount_ratio

        self.value_out, self.policy_out = self.network(num_units, num_layers, activation, is_training)

        self.value_loss = self.value_network_loss(self.value_out)
        self.policy_loss = self.policy_network_loss(self.policy_out)

        self.learning_rate = learning_rate
        self.opt = self.get_optimizer()

        self.action_softmax = tf.nn.softmax(self.policy_out)
        self.total_loss = self.get_total_loss(self.value_loss, self.policy_loss)
        with tf.name_scope('Loss'):
             tf.summary.scalar('loss', self.total_loss)

        self.min_opr = self.opt.minimize(self.total_loss)
        self.merged = tf.summary.merge_all()

    def setWriter(self, writer):
        self.writer = writer

    def network(self, num_units, num_layers, activation, is_training):

        with tf.variable_scope("mixed_network"):
            init_xavier = tf.contrib.layers.xavier_initializer()
            dense_out = layers.dense(self.s, units=num_units, activation=activation, kernel_initializer=init_xavier)
            for n in range(0, num_layers):
                dense_out = layers.dense(dense_out, units=num_units, activation=activation, kernel_initializer=init_xavier)
            policy_out = layers.dense(dense_out, units=self.a_len, activation=activation,
                                      kernel_initializer=init_xavier)
            value_out = layers.dense(dense_out, units=1, activation=activation, kernel_initializer=init_xavier)
            return value_out, policy_out

    def value_network_loss(self, value_out):
        return 0.5 * tf.reduce_mean(tf.square((value_out - self.v)))

    def policy_network_loss(self, policy_out):
        return -tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=policy_out, labels=self.a) * self.td_error)

    def get_total_loss(self, value_loss, policy_loss):
        return value_loss + policy_loss

    def mix_network_update(self, sess, episode_exp, step):
        episode_exp_size = len(episode_exp)
        s_ = np.zeros(shape=(episode_exp_size, self.s_len))
        s = np.zeros(shape=(episode_exp_size, self.s_len))
        r = np.zeros(shape=(episode_exp_size, 1))
        a = np.zeros(shape=episode_exp_size)

        for ind, exp in enumerate(episode_exp):
            s_[ind] = exp.current_state
            s[ind] = exp.last_state
            r[ind] = exp.reward
            a[ind] = exp.action
        v_ = self.get_value(sess, s_)
        v = r + self.d_r*self.d_r*v_
        feed_dict = {self.s: s, self.v: v,
                     self.td_error: self.get_td_error(sess,s_, s,r),
                     self.a: a}
        _, loss = sess.run([self.min_opr, self.total_loss], feed_dict)
        result = sess.run(self.merged, feed_dict= feed_dict)
        self.writer.add_summary(result, step)
        return loss

    def get_td_error(self,sess,s_,s,r):
       return r+self.d_r*self.get_value(sess,s_) - self.get_value(sess,s)

    def get_optimizer(self):
        return tf.train.AdamOptimizer(self.learning_rate)

    def update_network(self, sess, min_opt, feed_dict):
        sess.run(min_opt, feed_dict)

    def get_value(self, sess, s):
        return sess.run(self.value_out, feed_dict={self.s: s})

    def get_action(self, sess, s):
        return sess.run(self.policy_out, feed_dict={self.s: s})

    def choose_action(self, sess, s):
        s = np.reshape(s, newshape=(1, self.s_len))
        action, out = sess.run([self.action_softmax, self.policy_out], feed_dict={self.s: s})
        return np.random.choice(np.arange(action.shape[1]), p=action.ravel())

    def choose_action_nt(self, sess, s):
        s = np.reshape(s, newshape=(1, self.s_len))
        action, out = sess.run([self.action_softmax, self.policy_out], feed_dict={self.s: s})
        return np.argmax(action.ravel())
