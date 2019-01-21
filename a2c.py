import tensorflow as tf
from tensorflow import layers
import numpy as np


class A2C:

    def __init__(self, observe_space_len, action_space_len, discount_ratio, num_units, num_layers, activation,
                 learning_rate, n_step=1):

        with tf.Graph().as_default() as graph:
            self.s = tf.placeholder(tf.float64, shape=(None, observe_space_len), name="state")
            self.v = tf.placeholder(tf.float64, shape=None, name="value")
            self.td_error = tf.placeholder(tf.float64, shape=None, name="td_error")
            self.a = tf.placeholder(tf.int32, shape=None, name="action")

            self.s_len = observe_space_len
            self.a_len = action_space_len
            self.d_r = discount_ratio
            self.lr = learning_rate

            self.value_opr, self.policy_opr = self.get_network_opr(num_units, num_layers, activation)

            self.value_loss_opr = self.get_value_loss_opr(self.value_opr)
            self.policy_loss_opr = self.get_policy_loss_opr(self.policy_opr)

            self.optimizer_opr = self.get_optimizer_opr()

            self.softmax_policy_opr = tf.nn.softmax(self.policy_opr)
            self.total_loss_opr = self.get_total_loss_opr(self.value_loss_opr, self.policy_loss_opr)

            with tf.name_scope('Loss'):
                tf.summary.scalar('loss', self.total_loss_opr)

            self.min_opr = self.get_min_with_gc_opr(self.total_loss_opr)

            self.summary_opr = tf.summary.merge_all()

            self.global_step = tf.train.create_global_step()
            self.init_opr = tf.global_variables_initializer()
            self.saver_opr = tf.train.Saver()

            self.writer = tf.summary.FileWriter("TensorBoard/", graph=graph)
            self.graph = graph

            self.n_step = n_step

    ## Operators and Variables

    def get_min_with_gc_opr(self, loss):
        gvs = self.optimizer_opr.compute_gradients(loss)
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
        return self.optimizer_opr.apply_gradients(capped_gvs)

    def get_global_step_var(self):
        return self.global_step

    def get_summary_opr(self):
        return self.summary_opr

    def get_init_opr(self):
        return self.init_opr

    def get_saver_opr(self):
        return self.saver_opr

    def get_network_opr(self, num_units, num_layers, activation):

        init_xavier = tf.contrib.layers.xavier_initializer()
        dense_out = layers.dense(self.s, units=num_units, activation=activation, kernel_initializer=init_xavier)
        for n in range(0, num_layers):
            dense_out = layers.dense(dense_out, units=num_units, activation=activation, kernel_initializer=init_xavier)
        policy_out = layers.dense(dense_out, units=self.a_len, activation=activation,
                                  kernel_initializer=init_xavier)
        value_out = layers.dense(dense_out, units=1, activation=activation, kernel_initializer=init_xavier)
        return value_out, policy_out

    def get_value_loss_opr(self, value_out):
        return 0.5 * tf.reduce_mean(tf.square((value_out - self.v)))

    def get_policy_loss_opr(self, policy_out):
        return -tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=policy_out, labels=self.a) * self.td_error)

    def get_total_loss_opr(self, value_loss, policy_loss):
        return value_loss + policy_loss

    def get_optimizer_opr(self):
        return tf.train.AdamOptimizer(self.lr)

    ## Session Run

    def update_network(self, sess, min_opr, loss_opr, summary_opr, global_step, feed_dict):
        return sess.run([min_opr, loss_opr, summary_opr, global_step], feed_dict)

    def get_value(self, sess, s):
        return sess.run(self.value_opr, feed_dict={self.s: s})

    def get_action(self, sess, s):
        return sess.run(self.softmax_policy_opr, feed_dict={self.s: s})

    def get_td_error(self, sess, s, r):
        return r - self.get_value(sess, s)

    def get_old_td_error(self, sess, s, r, s_):
        return r + self.d_r * self.get_value(sess, s_) - self.get_value(sess, s)

    def get_td(self, sess, s_, r):
        return r + self.d_r * self.d_r * self.get_value(sess, s_)

    ## Learn

    def learn(self, sess, episode):
        experience = episode.experience
        experience_size = len(experience)

        last_state = experience[experience_size-1].current_state
        s = np.reshape(last_state, newshape=(1, self.s_len))
        last_state_value = self.get_value(sess, s)

        s = np.zeros(shape=(experience_size, self.s_len))
        s_= np.zeros(shape=(experience_size, self.s_len))
        r = np.zeros(shape=(experience_size, 1))
        a = np.zeros(shape=experience_size)
        t = np.zeros(shape=(experience_size, 1))
        r_ = np.zeros(shape=(experience_size, 1))

        for ind, exp in enumerate(experience):
            t[ind] = exp.reward
            if ind + self.n_step < experience_size:
                r[ind] = experience[ind + self.n_step].last_state_value
                r_[ind] = experience[ind + self.n_step].last_state_value
                for i in reversed(range(ind, ind + self.n_step)):
                    r[ind] = self.d_r * r[ind] + experience[i].reward
                    r_[ind] = self.d_r * self.d_r * r_[ind] + experience[i].reward
            else:
                r[ind] = last_state_value
                r_[ind] = last_state_value
                for i in reversed(range(ind, experience_size)):
                    r[ind] = self.d_r * r[ind] + experience[i].reward
                    r_[ind] = self.d_r * self.d_r * r_[ind] + experience[i].reward

            s_[ind] = exp.current_state
            s[ind] = exp.last_state
            a[ind] = exp.action

        print(r)
        print(self.get_value(sess, s))
        print(r == self.get_value(sess, s))

        print("----")
        print(np.array_equal(r_,self.get_td(sess, s_, t)))
        print(r_ == self.get_td(sess, s_, t))
        print(r_)
        print(self.get_td(sess, s_, t))
        print("--")
        print(np.array_equal(self.get_td_error(sess, s, r),  self.get_old_td_error(sess, s, t, s_)))
        print(self.get_td_error(sess, s, r) == self.get_old_td_error(sess, s, t, s_))
        print(self.get_td_error(sess, s, r))
        print(self.get_old_td_error(sess, s, t, s_))


        feed_dict = {self.s: s, self.v: r_,
                     self.td_error: self.get_td_error(sess, s, t),
                     self.a: a}
        _, loss, summary, global_step = self.update_network(sess, self.min_opr, self.total_loss_opr, self.summary_opr,
                                                            self.global_step, feed_dict)
        self.writer.add_summary(summary, global_step)
        return loss, global_step

    ## Choose Action

    def choose_action_with_exploration(self, sess, s):
        s = np.reshape(s, newshape=(1, self.s_len))
        action, value = sess.run([self.softmax_policy_opr, self.value_opr], feed_dict={self.s: s})
        return np.random.choice(np.arange(action.shape[1]), p=action.ravel()), value

    def choose_action(self, sess, s):
        s = np.reshape(s, newshape=(1, self.s_len))
        action, value = sess.run([self.softmax_policy_opr, self.value_opr], feed_dict={self.s: s})
        return np.argmax(action.ravel()), value
