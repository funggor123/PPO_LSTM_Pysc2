from model.feed_forward import Model
import tensorflow as tf
from tensorflow import layers


class ConvLSTM(Model):

    def __init__(self, a_len, a_dimension, obs_dimension, is_continuous, a_bound):
        super(ConvLSTM, self).__init__(a_len, a_dimension, obs_dimension, is_continuous, a_bound)
        self.lstm_unit = 64
        self.num_unit = 50
        self.greyscale = True

    def make_network(self, input_opr, name, train=True, reuse=False, batch_size=8):
        w_reg = tf.contrib.layers.l2_regularizer(self.L2_REG)
        with tf.variable_scope(name, reuse=reuse):
            if self.greyscale:
                input_opr = tf.image.rgb_to_grayscale(input_opr)
            conv1 = tf.layers.conv2d(inputs=input_opr, filters=8, kernel_size=4, strides=2, activation=tf.nn.relu6,
                                     trainable=train)
            conv2 = tf.layers.conv2d(inputs=conv1, filters=16, kernel_size=3, strides=2, activation=tf.nn.relu6,
                                     trainable=train)
            conv3 = tf.layers.conv2d(inputs=conv2, filters=32, kernel_size=3, strides=2, activation=tf.nn.relu6,
                                     trainable=train)
            conv4 = tf.layers.conv2d(inputs=conv3, filters=64, kernel_size=3, strides=2, activation=tf.nn.relu6,
                                     trainable=train)
            conv5 = tf.layers.conv2d(inputs=conv4, filters=128, kernel_size=3, strides=1, activation=tf.nn.relu6,
                                     trainable=train)
            conv6 = tf.layers.conv2d(inputs=conv5, filters=256, kernel_size=3, strides=1, activation=tf.nn.relu6,
                                     trainable=train)
            flatten = layers.flatten(inputs=conv6)

            state_in = tf.layers.flatten(flatten)

            # LSTM layer
            lstm = tf.nn.rnn_cell.LSTMCell(num_units=self.lstm_unit)
            init_state = lstm.zero_state(batch_size=batch_size, dtype=tf.float32)

            lstm_in = tf.expand_dims(state_in, axis=1)

            outputs, final_state = tf.nn.dynamic_rnn(cell=lstm, inputs=lstm_in, initial_state=init_state)
            cell_out = tf.reshape(outputs, [-1, self.lstm_unit], name='flatten_lstm_outputs')

            fc1 = layers.dense(cell_out, units=self.num_unit, activation=tf.nn.relu6,
                               trainable=train, kernel_regularizer=w_reg)

            fc2 = layers.dense(fc1, units=self.num_unit, activation=tf.nn.relu6,
                               trainable=train, kernel_regularizer=w_reg)

            value_out = self.value_output_layer(fc2, train, w_reg)

            if self.is_continuous:
                policy_out = self.continuous_policy_output_layer(fc2, train, w_reg)
            else:
                policy_out = self.discrete_policy_output_layer(fc2, train, w_reg)

            params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)

        return value_out, policy_out, params, init_state, final_state
