import tensorflow as tf
from tensorflow import layers


class Model:

    def __init__(self, a_len, a_dimension, obs_dimension, is_continuous, a_bound, is_cat=False):
        self.a_len = a_len
        self.a_bound = a_bound
        self.obs_dimension = obs_dimension
        self.a_dimension = a_dimension
        self.is_continuous = is_continuous
        self.num_unit = 50
        self.isCat = is_cat
        self.L2_REG = 0.001

    def discrete_policy_output_layer(self, fc1, train , w_reg):
        if self.isCat is True:
            a_logits = tf.layers.dense(fc1, 2, trainable=train, kernel_regularizer=w_reg)
            return tf.distributions.Categorical(logits=a_logits)
        else:
            return layers.dense(fc1, units=2, activation=tf.nn.softmax, kernel_regularizer=w_reg, trainable=train)

    def continuous_policy_output_layer(self, fc1, train, w_reg):
        log_sigma = tf.get_variable(name="pi_sigma", shape=self.a_dimension, initializer=tf.zeros_initializer(), trainable=train)
        bound = (self.a_bound[1] - self.a_bound[0]) / 2
        mu = tf.layers.dense(fc1, units=self.a_dimension[0], activation=tf.nn.tanh, trainable=train, kernel_regularizer=w_reg) * bound
        policy_out = tf.contrib.distributions.Normal(loc=mu, scale=tf.maximum(tf.exp(log_sigma), 0.0))
        return policy_out

    def value_output_layer(self, fc1, train, w_reg):
        value_out = layers.dense(fc1, units=1, trainable=train, kernel_regularizer=w_reg)
        return value_out

    def make_network(self, input_opr, name, train=True, reuse=False, batch_size=0):
        w_reg = tf.contrib.layers.l2_regularizer(self.L2_REG)
        with tf.variable_scope(name, reuse=reuse):
            fc1 = layers.dense(input_opr, units=self.num_unit, activation=tf.nn.relu6,
                               trainable=train, kernel_regularizer=w_reg)
            fc2 = layers.dense(fc1, units=self.num_unit, activation=tf.nn.relu6,
                               trainable=train, kernel_regularizer=w_reg)
            value_out = self.value_output_layer(fc2, train, w_reg)

            if self.is_continuous:
                policy_out = self.continuous_policy_output_layer(fc2, train, w_reg)
            else:
                policy_out = self.discrete_policy_output_layer(fc2, train, w_reg)

        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)

        return value_out, policy_out, params, None, None

    def make_actor_network(self, input_opr, name, train=True, reuse=False):

        with tf.variable_scope(name, reuse=reuse):
            fc1 = layers.dense(input_opr, units=200, activation=tf.nn.relu6,
                               trainable=train)
            if self.is_continuous:
                policy_out = self.continuous_policy_output_layer(fc1, train)
            else:
                policy_out = self.discrete_policy_output_layer(fc1, train)

        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return policy_out, params

    def make_critic_network(self, input_opr, name, train=True, reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            fc1 = layers.dense(input_opr, units=100, activation=tf.nn.relu6, trainable=train)
            value_out = self.value_output_layer(fc1, train)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return value_out, params
