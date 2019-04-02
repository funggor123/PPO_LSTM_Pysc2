from model.feed_forward import Model
import tensorflow as tf
from tensorflow import layers


class LSTM(Model):

    def __init__(self, a_len, a_dimension, obs_dimension, is_continuous, a_bound):
        super(LSTM, self).__init__(a_len, a_dimension, obs_dimension, is_continuous, a_bound)
        self.lstm_unit = 64

    def make_network(self, input_opr, name, train=True, reuse=False, batch_size=8):
        w_reg = tf.contrib.layers.l2_regularizer(self.L2_REG)
        with tf.variable_scope(name, reuse=reuse):

            lstm_in = tf.expand_dims(input_opr, axis=1)
            # LSTM layer
            lstm = tf.nn.rnn_cell.LSTMCell(num_units=self.lstm_unit)
            init_state = lstm.zero_state(batch_size=batch_size, dtype=tf.float32)

            outputs, final_state = tf.nn.dynamic_rnn(cell=lstm, inputs=lstm_in, initial_state=init_state)
            cell_out = tf.reshape(outputs, [-1, self.lstm_unit], name='flatten_lstm_outputs')

            fc1 = layers.dense(cell_out, units=50, activation=tf.nn.relu6,
                               trainable=train, kernel_regularizer=w_reg)

            fc2 = layers.dense(fc1, units=50, activation=tf.nn.relu6,
                               trainable=train, kernel_regularizer=w_reg)

            value_out = self.value_output_layer(fc2, train, w_reg)

            if self.is_continuous:
                policy_out = self.continuous_policy_output_layer(fc2, train, w_reg)
            else:
                policy_out = self.discrete_policy_output_layer(fc2, train, w_reg)

            params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)

        return value_out, policy_out, params, init_state, final_state
