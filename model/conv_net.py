from model.feed_forward import Model
import tensorflow as tf
from tensorflow import layers


class ConvNet(Model):

    def __init__(self, a_len, a_dimension, obs_dimension, is_continuous, a_bound):
        super(ConvNet, self).__init__(a_len, a_dimension, obs_dimension, is_continuous, a_bound)

    def make_network(self, input_opr, name, train=True):
        with tf.variable_scope(name):
            conv1 = tf.layers.conv2d(inputs=input_opr, filters=32, kernel_size=8, strides=4, activation=tf.nn.relu6, trainable=train)
            conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=4, strides=2, activation=tf.nn.relu6, trainable=train)
            conv3 = tf.layers.conv2d(inputs=conv2, filters=64, kernel_size=3, strides=1, activation=tf.nn.relu6, trainable=train)

            flatten = layers.flatten(inputs=conv3)

            fc1 = layers.dense(flatten, units=self.num_unit, activation=tf.nn.relu6,
                               trainable=train)
            value_out = self.value_output_layer(fc1, train)

            if self.is_continuous:
                policy_out = self.continuous_policy_output_layer(fc1, train)
            else:
                policy_out = self.discrete_policy_output_layer(fc1, train)

        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)

        return value_out, policy_out, params



