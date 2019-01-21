import reinforcement_learning as rl
import tensorflow as tf

actor, environment, train = rl.init()

with tf.Session(graph=actor.graph) as sess:
    if train is False:
        actor.get_saver_opr().restore(sess, "/tmp/model.ckpt")
    else:
        sess.run(tf.global_variables_initializer())
    while not train.stop():
        reward, loss, step = rl.train_episode(actor=actor, sess=sess, environment=environment,
                                              train=train.isTrain)
        train.add_total_loss(loss)
        train.add_total_reward(reward)
        train.add_total_episode()
        train.add_total_step(step)
        train.print_detail_in_every_episode(100, reward, actor)
        train.save_in_every_episode(100, sess, actor.get_saver_opr())


