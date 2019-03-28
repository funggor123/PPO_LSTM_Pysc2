import tensorflow as tf
from dist import config
from agent import agent
from gym import wrappers
import os
from tensorflow.python.training.summary_io import SummaryWriterCache

OUTPUT_RESULTS_DIR = "./"
ENVIRONMENT = 'Pong-v0'
SUMMARY_DIR = os.path.join(OUTPUT_RESULTS_DIR, "DPPO", ENVIRONMENT, "p")

class Worker:

    def __init__(self, FLAGS):
        self.worker, ps = config.readConfig("config.ini")
        self.cluster = tf.train.ClusterSpec({"ps": ps, "worker": self.worker})
        gpu_options = tf.GPUOptions(allow_growth=True)
        tf_config = tf.ConfigProto(gpu_options=gpu_options)
        self.server = tf.train.Server(self.cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index, config=tf_config)

        self.wid = FLAGS.task_index
        if FLAGS.job_name == "ps":
            self.server.join()
            print("--- Parameter Server Ready ---")
        elif FLAGS.job_name == "worker":
            self.nog = FLAGS.agg
            with tf.device(tf.train.replica_device_setter(
                    worker_device="/job:worker/task:%d" % FLAGS.task_index, cluster=self.cluster)):
                self.actor, self.environment, self.train = agent.init(self)

    def work(self):
        hooks = [self.actor.sync_replicas_hook]
        sess = tf.train.MonitoredTrainingSession(master=self.server.target, is_chief=(self.wid == 0),
                                                 save_summaries_steps=None, save_summaries_secs=None, hooks=hooks)

        if self.wid == 0:
            writer = SummaryWriterCache.get(SUMMARY_DIR)

        self.train.start(self.actor.get_saver_opr(), self.actor.get_init_opr(), sess, worker=True)
        while not sess.should_stop() and not (self.train.stop() and self.wid == 0):
            episode = agent.train_episode(actor=self.actor, sess=sess, environment=self.environment,
                                          train=self.train)
            self.train.add_episode(episode)
