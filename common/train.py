from common.episode import Episode
import tensorflow as tf


class Train:

    def __init__(self, train=True, max_episode=2000, print_every_episode=100, save_every_episode=200, batch_size=200,
                 max_step=1000):
        self.reward = 0
        self.loss = 0
        self.train = train
        self.num_episode = 0
        self.max_episode = max_episode
        self.print_every_episode = print_every_episode
        self.batch_size = batch_size
        self.max_step = max_step
        self.saver_opr = None
        self.init_opr = None
        self.sess = None
        self.save_every_episode = save_every_episode
        self.reward_list = []
        self.moving_average = None
        self.summaries = None
        self.writer = None
        self.worker = False

        self.moving_average = tf.Variable(1, name="w_1")
        tf.summary.scalar("normal/moving_mean", self.moving_average)
        self.summaries = tf.summary.merge_all()

    def add_episode(self, episode):
        self.num_episode = self.num_episode + 1
        self.reward = self.reward + episode.reward
        self.loss = episode.loss
        if len(self.reward_list) == 0:
            self.reward_list.append(self.reward)
        else:
            self.reward_list.append(self.reward_list[-1] * 0.9 + episode.reward * 0.1)
        summ = self.sess.run(self.summaries, feed_dict={ self.moving_average: self.reward_list[-1]})
        self.writer.add_summary(summ, global_step=self.num_episode)
        self.print_detail_in_every_episode(episode.reward)
        self.save_in_every_episode(self.save_every_episode, self.sess, self.saver_opr)

    def print_average_reward(self):
        print("Average Reward ", self.reward / self.num_episode)

    def print_loss(self):
        print("Loss ", self.loss)

    def print_num_episode(self, ):
        print("Episode ", self.num_episode)

    def print_detail_in_every_episode(self, reward):
        if self.num_episode % self.print_every_episode == 0:
            print("----------------------------")
            self.print_num_episode()
            self.print_average_reward()
            print("Ep Reward: ", reward)
            print("Moving Average Reward: ", self.reward_list[-1])
            self.print_loss()
            print("----------------------------")

    def stop(self):
        if self.num_episode >= self.max_episode:
            return True
        else:
            return False

    def stop_to_learn(self, current_step):
        if current_step == self.max_step - 1 or ((current_step % self.batch_size) == 0 and current_step is not 0):
            return True
        return False

    def start(self, saver_opr, init_opr, sess, worker=False):
        self.sess = sess
        self.worker = worker
        if self.train is False:
            self.saver_opr = saver_opr
            self.saver_opr.restore(sess, "/tmp/model.ckpt")
        else:
            self.saver_opr = saver_opr
            self.init_opr = init_opr
            self.writer = tf.summary.FileWriter("TensorBoard/", graph=sess.graph)
            if worker is False:
                self.sess.run(self.init_opr)

    def save_in_every_episode(self, num_of_episode, sess, saver):
        if self.num_episode % num_of_episode == 0:
            if self.worker is False:
                save_path = saver.save(sess, "/tmp/model.ckpt")
                print("Save in : ", save_path)
