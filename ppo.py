import tensorflow as tf
from tensorflow import layers
import numpy as np

from a2c import A2C


class PPO(A2C):

    def __init__(self, observe_space_len, action_space_len, discount_ratio, num_units, num_layers, activation,
                 learning_rate, n_step=1):
        super(PPO, self).__init__(id, observe_space_len, action_space_len, discount_ratio, num_units, num_layers, activation,
                 learning_rate)

    def learn(self, sess, episode):




    def get_updated_action_updated(self, sess, experience):
        vars = tf.trainable_variables()
        vars_ = sess.run(vars)
        action_prob_old = self.getActionProb(sess,experience.last_state, experience.action)
        super().learn(sess,experience)
        action_prob_new = self.getActionProb(sess, experience.last_state, experience.action)

        return self.get_likelihood_ratio(action_prob_old,action_prob_new)


    def getActionProb(self, sess, s, a):
        action = sess.run(self.softmax_policy_opr, feed_dict={self.s: s})
        return action[a]

    def get_likelihood_ratio(self, old, new):
        return new / old



