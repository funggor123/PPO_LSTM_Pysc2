import tensorflow as tf
import numpy as np


class A2C:

    def __init__(self, observe_space_len, action_space_dim, action_space_length, learning_rate, feature_transform,
                 model, regularization_stength):

        self.s_len = observe_space_len
        self.a_dim = action_space_dim
        self.lr = learning_rate
        self.model = model
        self.reg_s = regularization_stength
        self.action_space_length = action_space_length
        self.feature_transform = feature_transform

        self.s = tf.placeholder(tf.float32, shape=(None,) + observe_space_len, name="state")
        self.v = tf.placeholder(tf.float32, shape=(None, 1), name="value")
        self.td_error = tf.placeholder(tf.float32, shape=(None, 1), name="td_error")

        if model.is_continuous:
            self.a = tf.placeholder(tf.float32, shape=(None,) + action_space_dim, name="action")
        else:
            self.a = tf.placeholder(tf.int32, shape=[None, ], name="action")

        self.policy_opr, self.params = model.make_actor_network(input_opr=self.s,
                                                                name="target",
                                                                train=True)

        self.value_opr, self.value_params = model.make_critic_network(input_opr=self.s,
                                                      name="value",
                                                      train=True)
        with tf.variable_scope('value_loss'):
            self.value_loss_opr = self.__get_value_loss_opr__(self.value_opr, self.v)

        if model.is_continuous:
            with tf.variable_scope('continuous_policy_loss'):
                self.policy_loss_opr = self.__get_policy_continous_loss_opr__(self.policy_opr, self.a, self.td_error)
            with tf.variable_scope('sample_action'):
                self.out_opr = tf.squeeze(self.policy_opr.sample(1), axis=0)
                self.out_opr = tf.clip_by_value(self.out_opr, -2, 2)
        else:
            with tf.variable_scope('discrete_policy_loss'):
                self.policy_loss_opr = self.__get_policy_discrete_loss_opr__(self.policy_opr, self.a, self.td_error)
                self.out_opr = self.policy_opr
        with tf.variable_scope('total_loss'):
            self.total_loss_opr = self.__get_total_loss_opr__(self.value_loss_opr, self.policy_loss_opr)

        self.global_step = tf.train.create_global_step()
        self.optimizer_opr = self.__get_optimizer_opr__(self.lr)
        with tf.variable_scope('min_loss'):
            self.min_policy_loss_opr = self.__get_min_opr__(self.policy_loss_opr, self.optimizer_opr, self.global_step)
            self.min_value_loss_opr = self.__get_min_opr_without_gc__(self.value_loss_opr, self.optimizer_opr)
            self.min_total_loss_opr = self.__get_min_opr__(self.total_loss_opr, self.optimizer_opr,
                                                           self.global_step)
        '''
        self.summary_opr = tf.summary.merge_all()
        '''
        with tf.variable_scope('others'):
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
        exp = tf.squared_difference(value_out, v)
        return tf.reduce_mean(exp)

    def __get_policy_discrete_loss_opr__(self, policy_out, a, td_error):
        log_prob = tf.reduce_sum(tf.log(policy_out) * tf.one_hot(a, self.action_space_length[0], dtype=tf.float32),
                                 axis=1, keep_dims=True)
        exp_v = log_prob * td_error
        entropy = -tf.reduce_sum(policy_out * tf.log(policy_out), axis=1,
                                 keep_dims=True)
        aloss = self.reg_s * entropy + exp_v
        return tf.reduce_mean(-aloss)

    def __get_policy_continous_loss_opr__(self, policy_out, a, td_error):
        entropy = policy_out.entropy() * self.reg_s
        exp = policy_out.log_prob(a) * td_error
        return tf.reduce_mean(-exp) - tf.reduce_mean(entropy)

    def __get_total_loss_opr__(self, value_loss, policy_loss):
        return tf.add(value_loss, policy_loss)

    def __get_optimizer_opr__(self, lr):
        return tf.contrib.opt.NadamOptimizer(lr)

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
                                                       self.global_step, feed_dict)
        episode.loss = loss

        return episode, global_step

    ## Choose Action

    def choose_action(self, sess, s):
        shape = (1,) + self.s_len
        s = np.reshape(s, newshape=shape)
        action, value = sess.run([self.out_opr, self.value_opr], feed_dict={self.s: s})
        if self.model.is_continuous:
            action = np.reshape(action, newshape=self.a_dim)
            return action, value
        else:
            action = np.random.choice(range(action.shape[1]), p=action.ravel())
            return action, value
