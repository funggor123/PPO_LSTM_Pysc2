import reinforcement_learning as rl
import tensorflow as tf

actor, environment, train = rl.init()

with tf.Session() as sess:
    if train is False:
        actor.get_saver_opr().restore(sess, "/tmp/model.ckpt")
    else:
        sess.run(tf.global_variables_initializer())
    while not train.stop():
        episode, global_step = rl.train_episode(actor=actor, sess=sess, environment=environment, train=train.isTrain)
        train.set_loss(episode.loss)
        train.add_reward(episode.reward)
        train.add_episode()
        train.print_detail_in_every_episode(50, episode.reward)
        train.save_in_every_episode(100, sess, actor.get_saver_opr())
