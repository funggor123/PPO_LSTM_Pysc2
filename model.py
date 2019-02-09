import tensorflow as tf
from tensorflow import layers


class Model:

    def __init__(self, num_units, num_layers, activation, a_len, a_dim, o_len, is_continuous, a_bound):
        self.num_units = num_units
        self.num_layers = num_layers
        self.activation = activation
        self.a_len = a_len
        self.a_bound = a_bound
        self.o_len = o_len
        self.a_dim = a_dim
        self.is_continuous = is_continuous

    def discrete_output(self, fc1, train, init):
        return layers.dense(fc1, units=2, activation=tf.nn.softmax, kernel_initializer=init)

    def continous_output(self, fc1, train, init):
        sigma = layers.dense(fc1, units=1, activation=tf.nn.softplus,
                             trainable=train) + 1e-4
        mu = layers.dense(fc1, units=1, activation=tf.nn.tanh,
                          trainable=train) * self.a_bound[1]
        policy_out = tf.contrib.distributions.Normal(loc=mu, scale=sigma)
        return policy_out

    def value_output(self, fc1, train, init):
        value_out = layers.dense(fc1, units=1, kernel_initializer=init, trainable=train)
        return value_out

    def make_actor_network(self, input_opr, name, train=True):
        w_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope(name):
            dense_out = layers.dense(input_opr, units=100, activation=tf.nn.relu6,
                                     trainable=train, kernel_initializer=w_init)
            if self.is_continuous:
                policy_out = self.continous_output(dense_out, train, None)
            else:
                policy_out = self.discrete_output(dense_out, train, None)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return policy_out, params

    def make_critic_network(self, input_opr, name, train=True):
        w_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope(name):
            dense_out = layers.dense(input_opr, units=100, activation=tf.nn.relu6, kernel_initializer=w_init)
            value_out = self.value_output(dense_out, train, None)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return value_out, params


    '''
        def make_network(self, input_opr, name, train=True):
        with tf.variable_scope(name):
            init = tf.random_normal_initializer(0., .1)
            dense_out = layers.dense(input_opr, units=self.num_units, activation=self.activation,
                                     kernel_initializer=init,
                                     trainable=train)
            for n in range(0, self.num_layers - 1):
                dense_out = layers.dense(dense_out, units=self.num_units, activation=self.activation,
                                         kernel_initializer=init, trainable=train)
            if self.is_continuous:
                policy_out = self.continous_output(dense_out, train, init)
            else:
                policy_out = self.discrete_output(dense_out, train, init)

            value_out = self.value_output(dense_out, train, init)

        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)

        return value_out, policy_out, params
    '''
