import tensorflow as tf
import numpy as np
from pysc2.lib import actions
import common.utils as U


class Py_A2C:

    def __init__(self, msize, ssize, lr, feature_transform, model, regular_str, minibatch, epoch, isa2c=True, training=True):

        self.lr = lr
        self.model = model
        self.reg_str = regular_str
        self.feature_t = feature_transform

        self.training = training
        self.msize = msize
        self.ssize = ssize
        self.isize = len(actions.FUNCTIONS)
        self.epsilon = [0.05, 0.2]

        self.v = tf.placeholder(tf.float32, shape=(None, 1), name="value")
        self.td_error = tf.placeholder(tf.float32, shape=(None, 1), name="td_error")

        self.valid_spatial_action = tf.placeholder(tf.float32, [None], name='valid_spatial_action')
        self.spatial_action_selected = tf.placeholder(tf.float32, [None, self.ssize ** 2],
                                                      name='spatial_action_selected')
        self.valid_non_spatial_action = tf.placeholder(tf.float32, [None, len(actions.FUNCTIONS)],
                                                       name='valid_non_spatial_action')
        self.non_spatial_action_selected = tf.placeholder(tf.float32, [None, len(actions.FUNCTIONS)],
                                                          name='non_spatial_action_selected')
        self.learning_rate = tf.placeholder(tf.float32, shape=None, name="lr")

        # Set inputs of networks
        self.minimap = tf.placeholder(tf.float32, [None, U.minimap_channel(), self.msize, self.msize], name='minimap')
        self.screen = tf.placeholder(tf.float32, [None, U.screen_channel(), self.ssize, self.ssize], name='screen')
        self.info = tf.placeholder(tf.float32, [None, self.isize], name='info')

        self.input_opr = {"minimap": self.minimap, "screen": self.screen,
                          "info": self.info}

        if isa2c is False:
            self.dataset = tf.data.Dataset.from_tensor_slices({
                "v": self.v,
                "advantage": self.td_error,
                "valid_spatial_action": self.valid_spatial_action,
                "spatial_action_selected": self.spatial_action_selected,
                "valid_non_spatial_action": self.valid_non_spatial_action,
                "non_spatial_action_selected": self.non_spatial_action_selected,
                "minimap": self.minimap,
                "screen": self.screen,
                "info": self.info,
                "rewards": self.v})

            self.dataset = self.dataset.shuffle(buffer_size=10000)
            self.dataset = self.dataset.batch(minibatch)
            self.dataset = self.dataset.cache()
            self.dataset = self.dataset.repeat(epoch)
            self.iterator = self.dataset.make_initializable_iterator()
            self.batch = self.iterator.get_next()

            self.input_opr_batch = {"minimap": self.batch['minimap'], "screen": self.batch['screen'],
                                    "info": self.batch["info"]}

            self.value_out, self.policy_out, self.params, _, _ = model.make_network(
                input_opr=self.input_opr_batch ,
                name="target",
                num_action=self.isize,
                reuse=False)

            self.spatial_action_out = self.policy_out["spatial_action"]
            self.non_spatial_action_out = self.policy_out["non_spatial_action"]

        else:
            self.value_out, self.policy_out, self.params, _, _ = model.make_network(
                input_opr=self.input_opr,
                name="target",
                num_action=self.isize,
                reuse=False)

            self.spatial_action_out = self.policy_out["spatial_action"]
            self.non_spatial_action_out = self.policy_out["non_spatial_action"]

            self.value_eval, self.policy_eval, _, _, _ = model.make_network(self.input_opr,
                                                                            'target',
                                                                            num_action=self.isize,
                                                                            reuse=True)
            self.spatial_action_eval = self.policy_eval["spatial_action"]
            self.non_spatial_action_eval = self.policy_eval["non_spatial_action"]

        self.value_loss = self.get_value_loss(self.value_out, self.v)
        self.policy_loss = self.get_loss(self.get_log(self.spatial_action_out, self.spatial_action_selected, self.non_spatial_action_selected
                                                           , self.valid_non_spatial_action,
                                                           self.non_spatial_action_out,
                                                           self.valid_spatial_action
                                                           ), self.td_error)

        self.policy = [self.spatial_action_out, self.non_spatial_action_out]

        self.total_loss = self.get_total_loss(self.value_loss, self.policy_loss)

        self.global_step = tf.train.create_global_step()
        self.optimizer = self.get_optimizer(self.learning_rate)

        self.min_total_loss = self.get_min(self.total_loss, self.optimizer,
                                           self.global_step)
        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()


    def get_loss(self, log_prob, td_error):
        return tf.reduce_mean(-(log_prob * td_error))

    def get_log(self, spatial_action_out, spatial_action_selected, non_spatial_action_selected,valid_non_spatial_action, non_spatial_action_out, valid_spatial_action):
        spatial_action_prob = tf.reduce_sum(spatial_action_out * spatial_action_selected, axis=1)
        spatial_action_log_prob = tf.log(tf.clip_by_value(spatial_action_prob, 1e-10, 1.))
        non_spatial_action_prob = tf.reduce_sum(non_spatial_action_out * non_spatial_action_selected, axis=1)
        valid_non_spatial_action_prob = tf.reduce_sum(non_spatial_action_out * valid_non_spatial_action,
                                                      axis=1)
        valid_non_spatial_action_prob = tf.clip_by_value(valid_non_spatial_action_prob, 1e-10, 1.)
        non_spatial_action_prob = non_spatial_action_prob / valid_non_spatial_action_prob
        non_spatial_action_log_prob = tf.log(tf.clip_by_value(non_spatial_action_prob, 1e-10, 1.))
        return valid_spatial_action * spatial_action_log_prob + non_spatial_action_log_prob

    def get_global_step(self):
        return self.global_step

    def get_init_opr(self):
        return self.init

    def get_saver_opr(self):
        return self.saver

    def get_min(self, loss, opt_opr, global_step):
        return opt_opr.minimize(loss, global_step=global_step)

    def get_min_without_clip(self, loss, opt_opr):
        return opt_opr.minimize(loss)

    def ClipIfNotNone(self, grad):
        if grad is None:
            return grad
        return tf.clip_by_value(grad, -1, 1)

    def get_min_with_global_step(self, loss, opt_opr, global_step):
        gvs = opt_opr.compute_gradients(loss)

        clipped_gradients = [(self.ClipIfNotNone(grad), var) for grad, var in gvs]
        return opt_opr.apply_gradients(clipped_gradients, global_step=global_step)

    def get_min_with_clip_global_step(self, loss, opt_opr):
        gvs = opt_opr.compute_gradients(loss)
        clipped_gradients = [(self.ClipIfNotNone(grad), var) for grad, var in gvs]
        return opt_opr.apply_gradients(clipped_gradients)

    def get_value_loss(self, value_out, v):
        exp = tf.squared_difference(value_out, v)
        return tf.reduce_mean(exp)

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

    def get_value(self, sess, obs):
        minimap = np.array(obs.observation['feature_minimap'], dtype=np.float32)
        minimap = np.expand_dims(U.preprocess_minimap(minimap), axis=0)
        screen = np.array(obs.observation['feature_screen'], dtype=np.float32)
        screen = np.expand_dims(U.preprocess_screen(screen), axis=0)
        # TODO: only use available actions
        info = np.zeros([1, self.isize], dtype=np.float32)
        info[0, obs.observation['available_actions']] = 1

        feed = {self.minimap: minimap,
                self.screen: screen,
                self.info: info}
        value = sess.run(self.value_eval, feed_dict=feed)
        return value

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

        _, loss, global_step = self.update(sess, self.min_total_loss, self.total_loss,
                                           self.global_step, feed_dict)

        return loss

    def choose_action(self, sess, obs):
        minimap = np.array(obs.observation['feature_minimap'], dtype=np.float32)
        minimap = np.expand_dims(U.preprocess_minimap(minimap), axis=0)
        screen = np.array(obs.observation['feature_screen'], dtype=np.float32)
        screen = np.expand_dims(U.preprocess_screen(screen), axis=0)
        # TODO: only use available actions
        info = np.zeros([1, self.isize], dtype=np.float32)
        info[0, obs.observation['available_actions']] = 1

        feed = {self.minimap: minimap,
                self.screen: screen,
                self.info: info }

        non_spatial_action, spatial_action, value = sess.run(
            [self.non_spatial_action_eval, self.spatial_action_eval, self.value_out],
            feed_dict=feed)

        # Select an action and a spatial target
        non_spatial_action = non_spatial_action.ravel()
        spatial_action = spatial_action.ravel()
        valid_actions = obs.observation['available_actions']
        act_id = valid_actions[np.argmax(non_spatial_action[valid_actions])]
        target = np.argmax(spatial_action)
        target = [int(target // self.ssize), int(target % self.ssize)]

        # Epsilon greedy exploration
        if self.training and np.random.rand() < self.epsilon[0]:
            act_id = np.random.choice(valid_actions)
        if self.training and np.random.rand() < self.epsilon[1]:
            dy = np.random.randint(-4, 5)
            target[0] = int(max(0, min(self.ssize - 1, target[0] + dy)))
            dx = np.random.randint(-4, 5)
            target[1] = int(max(0, min(self.ssize - 1, target[1] + dx)))

        # Set act_id and act_args
        act_args = []
        for arg in actions.FUNCTIONS[act_id].args:
            if arg.name in ('screen', 'minimap', 'screen2'):
                act_args.append([target[1], target[0]])
            else:
                act_args.append([0])
        return act_id, act_args, value

