from model.feed_forward import Model
import tensorflow as tf
from tensorflow import layers


class ConvNet(Model):

    def __init__(self, a_len, a_dimension, obs_dimension, is_continuous, a_bound):
        super(ConvNet, self).__init__(a_len, a_dimension, obs_dimension, is_continuous, a_bound)

    def make_network(self, input_opr, name, train=True):
        with tf.variable_scope(name):
            init_xavier = tf.contrib.layers.xavier_initializer()

            conv1 = layers.conv2d(inputs=input_opr, filters=6, kernel_size=(15, 15), strides=(1, 1),
                                  activation=tf.nn.relu6, kernel_initializer=init_xavier,
                                  trainable=train)
            pool1 = layers.max_pooling2d(inputs=conv1, pool_size=(2, 2), strides=(2, 2))

            conv2 = layers.conv2d(inputs=pool1, filters=16, kernel_size=(5, 5), strides=(1, 1),
                                  activation=tf.nn.relu6,
                                  kernel_initializer=init_xavier,
                                  trainable=train)

            pool2 = layers.max_pooling2d(inputs=conv2, pool_size=(2, 2), strides=(1, 1))

            conv3 = layers.conv2d(inputs=pool2, filters=120, kernel_size=(5, 5), strides=(1, 1),
                                  activation=tf.nn.relu6,
                                  kernel_initializer=init_xavier,
                                  trainable=train)

            flatten = layers.flatten(inputs=conv3)

            fc1 = layers.dense(flatten, 84, activation=tf.nn.relu6, kernel_initializer=init_xavier,
                               trainable=train)

            if self.is_continuous:
                policy_out, value_out = self.continuous_policy_output_layer(fc1, train, init_xavier)
            else:
                policy_out, value_out = self.discrete_policy_output_layer(fc1, train, init_xavier)

        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)

        return value_out, policy_out, params, fc1


