from model import Model
import tensorflow as tf
from tensorflow import layers


class ConvNet(Model):

    def __init__(self, activation, a_len, o_len, is_continuous):
        super(ConvNet, self).__init__(1, 40, activation, a_len, o_len, is_continuous)

    def make_network(self, input_opr, name, train=True):
        with tf.variable_scope(name):
            init_xavier = tf.contrib.layers.xavier_initializer()

            conv1 = layers.conv2d(inputs=input_opr, filters=6, kernel_size=(15, 15), strides=(1, 1),
                                  activation=self.activation, kernel_initializer=init_xavier,
                                  trainable=train)
            pool1 = layers.max_pooling2d(inputs=conv1, pool_size=(2,2), strides=(2,2))

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

            value_out = layers.dense(fc1, units=1, activation=self.activation, kernel_initializer=init_xavier,
                                     trainable=train)

            mu1 = layers.dense(fc1, self.a_len, activation="tanh", kernel_initializer=init_xavier,
                              trainable=train)

            mu2 = layers.dense(fc1, self.a_len, activation="softmax", kernel_initializer=init_xavier,
                               trainable=train)

            mu1 = mu1 * 1

            mu2 = mu2 * 1

            concat = tf.concat([mu1, mu2], 1)

            sigma = layers.dense(fc1, self.a_len, activation="softplus", kernel_initializer=init_xavier,
                                 trainable=train)
            sigma = sigma + 1e-4

            with tf.control_dependencies([concat]):
                policy_out = tf.contrib.distributions.Normal(mu1, sigma)

        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)

        return value_out, policy_out, params, sigma

    def discrete_output(self, flatten, train, init):
        policy_out = layers.dense(flatten, units=self.a_len, activation=self.activation,
                                  kernel_initializer=init, trainable=train)
        return policy_out
