import tensorflow as tf
from tensorflow import layers


class Model:

    def __init__(self, num_units, num_layers, activation, a_len, o_len, is_continuous):
        self.num_units = num_units
        self.num_layers = num_layers
        self.activation = activation
        self.a_len = a_len
        self.o_len = o_len
        self.is_continuous = is_continuous

    def make_network(self, input_opr, name, train=True):
        with tf.variable_scope(name):
            init_xavier = tf.contrib.layers.xavier_initializer()
            dense_out = layers.dense(input_opr, units=self.num_units, activation=self.activation, kernel_initializer=init_xavier,
                                     trainable=train)
            for n in range(0, self.num_layers-1):
                dense_out = layers.dense(dense_out, units=self.num_units, activation=self.activation,
                                         kernel_initializer=init_xavier, trainable=train)
            policy_out = layers.dense(dense_out, units=self.a_len, activation=self.activation,
                                      kernel_initializer=init_xavier, trainable=train)
            value_out = layers.dense(dense_out, units=1, activation=self.activation, kernel_initializer=init_xavier,
                                     trainable=train)

        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)

        return value_out, policy_out, params
