class Train:

    def __init__(self, is_train=True, max_episode=2000):
        self.reward = 0
        self.loss = 0
        self.isTrain = is_train
        self.episode = 0
        self.max_episode = max_episode

    def add_reward(self, reward):
        self.reward = self.reward + reward

    def set_loss(self, loss):
        self.loss = loss

    def add_episode(self):
        self.episode = self.episode + 1

    def print_total_reward(self):
        print("Average Reward ", self.reward / self.episode)

    def print_loss(self):
        print("Loss ", self.loss)

    def print_episode(self, ):
        print("Episode ", self.episode)

    def print_detail_in_every_episode(self, num_of_episode, reward):
        if self.episode % num_of_episode == 0:
            self.print_episode()
            self.print_total_reward()
            self.print_loss()
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
