import tensorflow as tf
import numpy as np


class A2C:

    def __init__(self, obs_dimension, a_dimension, action_space_length, lr, feature_transform,
                 model, regular_str, minibatch, epoch, max_grad_norm, isLSTM, isa2c=True, is_seperate=False):

        self.obs_dim = obs_dimension
        self.a_dim = a_dimension
        self.lr = lr
        self.model = model
        self.reg_str = regular_str
        self.action_space_length = action_space_length
        self.feature_t = feature_transform
        self.is_seperate = is_seperate
        self.max_grad_norm = max_grad_norm
        self.isLSTM = isLSTM

        self.s = tf.placeholder(tf.float32, shape=(None,) + obs_dimension, name="state")
        self.v = tf.placeholder(tf.float32, shape=(None, 1), name="value")
        self.td_error = tf.placeholder(tf.float32, shape=(None, 1), name="td_error")

        if model.is_continuous:
            self.a = tf.placeholder(tf.float32, shape=(None,) + a_dimension, name="action")
        else:
            self.a = tf.placeholder(tf.int32, shape=[None, ], name="action")

        if isa2c is False:
            if isLSTM:
                self.dataset = tf.data.Dataset.from_tensor_slices({"state": self.s, "actions": self.a,
                                                                   "rewards": self.v, "advantage": self.td_error})
                self.dataset = self.dataset.batch(minibatch, drop_remainder=True)
                self.dataset = self.dataset.cache()
                self.dataset = self.dataset.repeat(epoch)
                self.dataset = self.dataset.shuffle(buffer_size=10000)
                self.iterator = self.dataset.make_initializable_iterator()
                self.batch = self.iterator.get_next()

            else:
                self.dataset = tf.data.Dataset.from_tensor_slices({"state": self.s, "actions": self.a,
                                                                   "rewards": self.v, "advantage": self.td_error})
                self.dataset = self.dataset.shuffle(buffer_size=10000)
                self.dataset = self.dataset.batch(minibatch)
                self.dataset = self.dataset.cache()
                self.dataset = self.dataset.repeat(epoch)
                self.iterator = self.dataset.make_initializable_iterator()
                self.batch = self.iterator.get_next()

            if self.is_seperate:
                self.policy_out, self.params = model.make_actor_network(input_opr=self.batch['state'],
                                                                        name="target",
                                                                        train=True,
                                                                        batch_size=minibatch
                                                                        )
                self.value_out, self.value_params = model.make_critic_network(input_opr=self.batch['state'],
                                                                              name="target_value",
                                                                              train=True,
                                                                              batch_size=minibatch
                                                                              )
            else:
                self.value_out, self.policy_out, self.params, self.i_state, self.f_state = model.make_network(
                    input_opr=self.batch['state'],
                    name="target",
                    batch_size=minibatch,
                    train=True)
        else:
            self.value_out, self.policy_out, self.params, self.old_i_state, self.old_f_state = model.make_network(
                input_opr=self.s,
                name="target",
                batch_size=minibatch,
                train=True)

        if self.is_seperate:
            self.policy_eval, _ = model.make_actor_network(self.s, 'target', batch_size=1, reuse=True)
            self.value_eval, _ = model.make_critic_network(self.s, 'target_value', batch_size=1, reuse=True)
        else:
            self.value_eval, self.policy_eval, _, self.eval_i_state, self.eval_f_state = model.make_network(self.s,
                                                                                                            'target',
                                                                                                            reuse=True,
                                                                                                            batch_size=1)

        self.value_loss = self.get_value_loss(self.value_out, self.v)

        if model.is_continuous:
            self.policy_loss = self.get_con_policy_loss(self.policy_out, self.a, self.td_error)
            self.policy = tf.squeeze(self.policy_eval.sample(1), axis=0)
        else:
            self.policy_loss = self.get_discrete_policy_loss(self.policy_out, self.a, self.td_error)
            if self.model.isCat:
                self.policy = tf.squeeze(self.policy_eval.sample(1), axis=0)
            else:
                self.policy = self.policy_eval

        self.total_loss = self.get_total_loss(self.value_loss, self.policy_loss)

        self.global_step = tf.train.create_global_step()
        self.optimizer = self.get_optimizer(self.lr)

        self.min_policy_loss = self.get_min_clip(self.policy_loss, self.optimizer)
        self.min_value_loss = self.get_min_clip(self.value_loss, self.optimizer)
        self.min_total_loss = self.get_min_clip(self.total_loss, self.optimizer)

        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

    def get_global_step(self):
        return self.global_step

    def get_init_opr(self):
        return self.init

    def get_saver_opr(self):
        return self.saver

    def get_min_global_step(self, loss, opt_opr, global_step):
        return opt_opr.minimize(loss, global_step=global_step)

    def get_min(self, loss, opt_opr):
        return opt_opr.minimize(loss)

    def get_min_clip_global_step(self, loss, opt_opr, global_step):
        gradients, variables = zip(*opt_opr.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, self.max_grad_norm)
        return opt_opr.apply_gradients(zip(gradients, variables), global_step)

    def get_min_clip(self, loss, opt_opr):
        gradients, variables = zip(*opt_opr.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, self.max_grad_norm)
        return opt_opr.apply_gradients(zip(gradients, variables))

    def get_value_loss(self, value_out, v):
        exp = tf.squared_difference(value_out, v)
        return tf.reduce_mean(exp)

    def get_discrete_policy_loss(self, policy_out, a, td_error):
        if self.model.isCat:
            entropy = policy_out.entropy()
            loss = policy_out.log_prob(a) * td_error
            return tf.reduce_mean(-loss) + tf.reduce_mean(-entropy) * self.reg_str
        else:
            entropy = -tf.reduce_sum(policy_out * tf.log(policy_out), axis=1,
                                     keepdims=True)
            log_prob = tf.reduce_sum(tf.log(policy_out) * tf.one_hot(a, self.action_space_length[0], dtype=tf.float32),
                                     axis=1, keepdims=True)
            loss = log_prob * td_error
            return tf.reduce_mean(-loss) + tf.reduce_mean(-entropy) * self.reg_str

    def get_con_policy_loss(self, policy_out, a, td_error):
        entropy = policy_out.entropy()
        loss = policy_out.log_prob(a) * td_error
        return tf.reduce_mean(-loss) + tf.reduce_mean(-entropy) * self.reg_str

    def get_total_loss(self, value_loss, policy_loss):
        return tf.add(value_loss, policy_loss)

    def get_optimizer(self, lr):
        return tf.contrib.opt.NadamOptimizer(lr)

    def update(self, sess, min_opr, loss_opr, global_step, feed_dict):
        return sess.run([min_opr, loss_opr, global_step], feed_dict)

    def update_without_global_step(self, sess, min_opr, loss_opr, feed_dict):
        return sess.run([min_opr, loss_opr], feed_dict)

    def get_computed_gradient(self, loss, opt):
        return opt.compute_gradients(loss)

    def get_value(self, sess, s, final_state):
        if self.isLSTM:
            s = np.reshape(s, newshape=(1,) + self.obs_dim)
            value = sess.run(self.value_eval, feed_dict={self.s: s, self.eval_i_state: final_state})
        else:
            s = np.reshape(s, newshape=(1,) + self.obs_dim)
            value = sess.run(self.value_eval, feed_dict={self.s: s})
        return value

    def learn(self, sess, episode):
        s, s_, a, r, v, g_adv, adv, q, experience_size = self.feature_t.transform(episode)
        feed_dict = {self.s: s,
                     self.td_error: g_adv,
                     self.a: a,
                     self.v: q
                     }

        _, loss, global_step = self.update(sess, self.min_total_loss, self.total_loss,
                                           self.global_step, feed_dict)
        episode.loss = loss

        return episode

    def get_init_state(self, sess):
        return sess.run(self.eval_i_state)

    def choose_action(self, sess, s, i_state):
        if self.isLSTM:
            shape = (1,) + self.obs_dim
            s = np.reshape(s, newshape=shape)
            action, value, f_state = sess.run([self.policy, self.value_eval, self.eval_f_state],
                                              feed_dict={self.s: s, self.eval_i_state: i_state})
            if self.model.is_continuous:
                action = np.clip(action, self.model.a_bound[0], self.model.a_bound[1])
                action = np.reshape(action, newshape=self.a_dim)
                return action, value, f_state
            else:
                if self.model.isCat:
                    action = action[0]
                else:
                    action = np.random.choice(range(action.shape[1]), p=action.ravel())
                return action, value, f_state
        else:
            shape = (1,) + self.obs_dim
            s = np.reshape(s, newshape=shape)
            action, value = sess.run([self.policy, self.value_eval],
                                     feed_dict={self.s: s})
            if self.model.is_continuous:
                action = np.clip(action, self.model.a_bound[0], self.model.a_bound[1])
                action = np.reshape(action, newshape=self.a_dim)
                return action, value, None
            else:
                if self.model.isCat:
                    action = action[0]
                else:
                    action = np.random.choice(range(action.shape[1]), p=action.ravel())
                return action, value, None
