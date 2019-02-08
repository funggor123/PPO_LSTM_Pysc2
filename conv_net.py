from model import Model
import tensorflow as tf
from tensorflow import layers


class ConvNet(Model):

    def __init__(self, activation, a_len, a_dim, o_len, is_continuous, a_bound):
        super(ConvNet, self).__init__(1, 40, activation, a_len, a_dim, o_len, is_continuous, a_bound)

    def make_network(self, input_opr, name, train=True):
        with tf.variable_scope(name):
            init_xavier = tf.contrib.layers.xavier_initializer()

            conv1 = layers.conv2d(inputs=input_opr, filters=6, kernel_size=(15, 15), strides=(1, 1),
                                  activation=self.activation, kernel_initializer=init_xavier,
                                  trainable=train)
            pool1 = layers.max_pooling2d(inputs=conv1, pool_size=(2, 2), strides=(2, 2))

            conv2 = layers.conv2d(inputs=pool1, filters=16, kernel_size=(5, 5), strides=(1, 1),
                                  activation=self.activation,
                                  kernel_initializer=init_xavier,
                                  trainable=train)

            pool2 = layers.max_pooling2d(inputs=conv2, pool_size=(2, 2), strides=(1, 1))

            conv3 = layers.conv2d(inputs=pool2, filters=120, kernel_size=(5, 5), strides=(1, 1),
                                  activation=self.activation,
                                  kernel_initializer=init_xavier,
                                  trainable=train)

            flatten = layers.flatten(inputs=conv3)

            fc1 = layers.dense(flatten, 84, activation=self.activation, kernel_initializer=init_xavier,
                               trainable=train)

            if self.is_continuous:
                policy_out, value_out = self.continous_output(fc1, train, init_xavier)
            else:
                policy_out, value_out = self.discrete_output(fc1, train, init_xavier)

        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)

        return value_out, policy_out, params, fc1

    def discrete_output(self, fc1, train, init):
        policy_out = []
        for val in self.a_len:
            policy_out.append(layers.dense(fc1, units=val, activation=self.activation,
                                           kernel_initializer=init, trainable=train))

        value_out = layers.dense(fc1, units=1, activation=self.activation, kernel_initializer=init,
                                 trainable=train)
        return value_out, policy_out

    def continous_output(self, fc1, train, init):
        value_out = layers.dense(fc1, units=1, activation=self.activation, kernel_initializer=init,
                                 trainable=train)
        mu = layers.dense(fc1, self.a_dim, activation="tanh", kernel_initializer=init,
                          trainable=train)
        sigma = layers.dense(fc1, self.a_dim, activation="softplus", kernel_initializer=init,
                             trainable=train)
        mu = mu * self.a_bound[1]
        sigma = sigma + 1e-4
        policy_out = tf.distributions.Normal(loc=tf.reduce_mean(mu), scale=tf.reduce_mean(sigma))

        return value_out, policy_out
