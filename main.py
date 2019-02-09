import tensorflow as tf
from agent import agent

actor, environment, train = agent.init()

with tf.Session() as sess:
    train.start(actor.get_saver_opr(), actor.get_init_opr(), sess)
    while not train.stop():
        episode = agent.train_episode(actor=actor, sess=train.sess, environment=environment, train=train)
        train.add_episode(episode)
