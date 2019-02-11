from model.feed_forward import Model
import tensorflow as tf
from tensorflow import layers


class ConvLSTM(Model):

    def __init__(self, a_len, a_dimension, obs_dimension, is_continuous, a_bound):
        super(ConvLSTM, self).__init__(a_len, a_dimension, obs_dimension, is_continuous, a_bound)
        self.lstm_unit = 256

    def make_network(self, input_opr, name, train=True, reuse=False, batch_size=0):
        with tf.variable_scope(name, reuse=reuse):
            conv1 = tf.layers.conv2d(inputs=input_opr, filters=32, kernel_size=8, strides=4, activation=tf.nn.relu)
            conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=4, strides=2, activation=tf.nn.relu)
            conv3 = tf.layers.conv2d(inputs=conv2, filters=64, kernel_size=3, strides=1, activation=tf.nn.relu)

            state_in = tf.layers.flatten(conv3)

            layer1 = tf.layers.dense(state_in, 400, tf.nn.relu, name="l1")

            layer2 = tf.layers.dense(layer1, 256, tf.nn.relu, name="l2")

            # LSTM layer
            lstm = tf.nn.rnn_cell.LSTMCell(num_units=256, name='basic_lstm_cell')
            lstm = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=1)
            lstm = tf.nn.rnn_cell.MultiRNNCell(cells=[lstm] * 1)

            init_state = lstm.zero_state(batch_size=batch_size, dtype=tf.float32)
            lstm_in = tf.expand_dims(layer2, axis=1)

            outputs, final_state = tf.nn.dynamic_rnn(cell=lstm, inputs=lstm_in, initial_state=init_state)
            cell_out = tf.reshape(outputs, [-1, 256], name='flatten_lstm_outputs')

            value_out = self.value_output_layer(cell_out, train)

            if self.is_continuous:
                policy_out = self.continuous_policy_output_layer(cell_out, train)
            else:
                policy_out = self.discrete_policy_output_layer(cell_out, train)

            params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)

        return value_out, policy_out, params, init_state, final_state
