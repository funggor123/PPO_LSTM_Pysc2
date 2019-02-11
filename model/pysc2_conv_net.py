from model.feed_forward import Model
import tensorflow as tf
import tensorflow.contrib.layers as layers


class ConvNet:

    def make_network(self, input_opr, name, num_action, reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            mconv1 = layers.conv2d(tf.transpose(input_opr["minimap"], [0, 2, 3, 1]),
                                   num_outputs=16,
                                   kernel_size=5,
                                   stride=1,
                                   scope='mconv1')
            mconv2 = layers.conv2d(mconv1,
                                   num_outputs=32,
                                   kernel_size=3,
                                   stride=1,
                                   scope='mconv2')
            sconv1 = layers.conv2d(tf.transpose(input_opr["screen"], [0, 2, 3, 1]),
                                   num_outputs=16,
                                   kernel_size=5,
                                   stride=1,
                                   scope='sconv1')
            sconv2 = layers.conv2d(sconv1,
                                   num_outputs=32,
                                   kernel_size=3,
                                   stride=1,
                                   scope='sconv2')
            info_fc = layers.fully_connected(layers.flatten(input_opr["info"]),
                                             num_outputs=256,
                                             activation_fn=tf.tanh,
                                             scope='info_fc')

            feat_conv = tf.concat([mconv2, sconv2], axis=3)
            spatial_action = layers.conv2d(feat_conv,
                                           num_outputs=1,
                                           kernel_size=1,
                                           stride=1,
                                           activation_fn=None,
                                           scope='spatial_action')
            spatial_action = tf.nn.softmax(layers.flatten(spatial_action))

            feat_fc = tf.concat([layers.flatten(mconv2), layers.flatten(sconv2), info_fc], axis=1)
            feat_fc = layers.fully_connected(feat_fc,
                                             num_outputs=256,
                                             activation_fn=tf.nn.relu,
                                             scope='feat_fc')
            non_spatial_action = layers.fully_connected(feat_fc,
                                                        num_outputs=num_action,
                                                        activation_fn=tf.nn.softmax,
                                                        scope='non_spatial_action')
            value = tf.reshape(layers.fully_connected(feat_fc,
                                                      num_outputs=1,
                                                      activation_fn=None,
                                                      scope='value'), [-1])

        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        print(spatial_action.shape)
        policy_out = {"spatial_action": spatial_action, "non_spatial_action": non_spatial_action}

        return value, policy_out, params, None, None
