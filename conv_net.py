from model import Model
import tensorflow as tf
from tensorflow import layers


class ConvNet(Model):

    def __init__(self, activation, a_len, o_len, is_continuous):
        super(ConvNet, self).__init__(1, 40, activation, a_len, o_len, is_continuous)

    def make_network(self, input_opr, name, train=True):
        with tf.variable_scope(name):
            init_xavier = tf.contrib.layers.xavier_initializer()

            conv1 = layers.conv2d(inputs=input_opr, filters=4, kernel_size=(8, 8), strides=(4, 4),
                                  activation=self.activation, kernel_initializer=init_xavier,
                                  trainable=train)

            conv2 = layers.conv2d(inputs=conv1, filters=2, kernel_size=(4, 4), strides=(2, 2),
                                  activation=self.activation,
                                  kernel_initializer=init_xavier,
                                  trainable=train)

            conv3 = layers.conv2d(inputs=conv2, filters=2, kernel_size=(3, 3), strides=(1, 1),
                                  activation=self.activation,
                                  kernel_initializer=init_xavier,
                                  trainable=train)

            flatten = layers.flatten(inputs=conv3)

            value_out = layers.dense(flatten, units=1, activation=self.activation, kernel_initializer=init_xavier,
                                     trainable=train)

            fc1 = layers.dense(flatten, 512, activation=self.activation, kernel_initializer=init_xavier,
                               trainable=train)

            mu = layers.dense(fc1, self.a_len, activation="tanh", kernel_initializer=init_xavier,
                              trainable=train)

            sigma = layers.dense(fc1, self.a_len, activation="softplus", kernel_initializer=init_xavier,
                                 trainable=train)

            policy_out = tf.distributions.Normal(loc=mu, scale=sigma)

        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)

        return value_out, policy_out, params, mu

    def discrete_output(self, flatten, train, init):
        policy_out = layers.dense(flatten, units=self.a_len, activation=self.activation,
                                  kernel_initializer=init, trainable=train)
        return policy_out
