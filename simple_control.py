import gym
import a2c as a2c
import tensorflow as tf
import experience as ep

train = False
left, stay, right = 0, 1, 2
action_space_len = 2
observation_space_len = 4

actor = a2c.A2C(num_units=30,
                num_layers=2,
                action_space_len=action_space_len,
                observe_space_len=observation_space_len,
                activation="elu",
                discount_ratio=0.90,
                learning_rate=0.005)

saver = tf.train.Saver()
sess = tf.Session()

if train is False:
    saver.restore(sess, "/tmp/model.ckpt")
else:
    sess.run(tf.global_variables_initializer())
if train is True:
    writer = tf.summary.FileWriter("TensorBoard/train/", graph=sess.graph)
    actor.setWriter(writer)
else:
    writer = tf.summary.FileWriter("TensorBoard/nontrain/", graph=sess.graph)
    actor.setWriter(writer)

env = gym.make('CartPole-v0')
train_reward = 0
total_loss = 0
ten_round = 0
for i_episode in range(7000):
    observation = env.reset()
    episode_exp = []
    round_r = 0
    for ind in range(300):
        if train is False:
            env.render()
        exp = ep.Experience(observation_space_len, action_space_len)
        exp.set_last_state(observation)
        if train is False:
            action = actor.choose_action_nt(sess, observation)
        else:
            action = actor.choose_action(sess, observation)
        observation, reward, done, info = env.step(action)
        exp.set_action(action)
        exp.set_reward(reward)
        train_reward += reward
        round_r += reward
        exp.set_current_state(observation)
        episode_exp.append(exp)
        if done:
            if train is True:
                loss = actor.mix_network_update(sess, episode_exp, i_episode)
                total_loss += loss
            break
    if i_episode % 100 == 0 and train is True:
        save_path = saver.save(sess, "/tmp/model.ckpt")
        print(round_r)
        print(train_reward / (i_episode + 1))
        print(total_loss / (i_episode + 1))
        print(i_episode)
        if round_r == 200:
            ten_round = ten_round + 1
        else:
            ten_round = 0
        if ten_round == 5:
            print(i_episode)
            break
