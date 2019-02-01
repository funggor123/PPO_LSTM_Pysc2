class Train:

    def __init__(self, is_train=True, max_episode=2000):
        self.total_reward = 0
        self.total_loss = 0
        self.isTrain = is_train
        self.episode = 0
        self.max_episode = max_episode

    def add_total_reward(self, reward):
        self.total_reward = self.total_reward + reward

    def add_total_loss(self, loss):
        self.total_loss = self.total_loss + loss

    def add_total_episode(self):
        self.episode = self.episode + 1

    def print_total_reward(self):
        print("Average Reward ", self.total_reward / self.episode)

    def print_total_loss(self):
        print("Average Loss ", self.total_loss / self.episode)

    def print_episode(self, ):
        print("Episode ", self.episode)

    def print_detail_in_every_episode(self, num_of_episode, reward, saver):
        if self.episode % num_of_episode == 0:
            self.print_episode()
            self.print_total_reward()
            self.print_total_loss()
            print("Episode Reward ", reward)

    def stop(self):
        if self.episode >= self.max_episode:
            return True
        else:
            return False

    def save_in_every_episode(self, num_of_episode, sess, saver):
        if self.episode % num_of_episode == 0:
            save_path = saver.save(sess, "/tmp/model.ckpt")
            print("Save in : ", save_path)
