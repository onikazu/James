import tensorflow as tf
import numpy as np
from ornstein_uhlenbeck_action_noise import OrnsteinUhlenbeckActionNoise


class Actor(object):
    def __init__(self, action_space, observation_space):
        observation_dim = observation_space.shape[0]
        action_dim = action_space.shape[0]
        self.action_space = action_space
        self.actor_noise = \
            OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))

        self.inputs = tf.placeholder(tf.float32, [None, observation_dim])
        self.fc1 = tf.contrib.layers.fully_connected(self.inputs, 400)
        self.fc2 = tf.contrib.layers.fully_connected(self.fc1, 300)
        self.fc3 = tf.contrib.layers.fully_connected(self.fc2, action_dim,
                                                     activation_fn=tf.tanh)
        self.action = tf.multiply(self.fc3, 1)

        self.network_params = tf.trainable_variables()
        self.action_gradient = tf.placeholder(tf.float32, [None, action_dim])

        # Combine the gradients here
        self.unnormalized_actor_gradients = tf.gradients(self.action,
                                                         self.network_params,
                                                         -self.action_gradient)
        self.actor_gradients = list(map(lambda x: tf.div(x, 20),
                                        self.unnormalized_actor_gradients))

        # Optimization Op
        self.opt = tf.train.AdamOptimizer(0.0001). \
            apply_gradients(zip(self.actor_gradients, self.network_params))

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        # network_param = self.sess.run(self.network_params)
        # print(network_param)
        # print(len(network_param))

    def get_action_for_train(self, state, ep):
        action = self.get_action(state) + self.actor_noise()
        return action

    def get_action(self, state):
        action = self.get_actions([state])
        action = action[0]
        return action

    def get_actions(self, states):
        feed = {self.inputs: states}
        actions = self.sess.run(self.action, feed_dict=feed)
        return actions

    def train(self, states, action_gradients):
        feed = {self.inputs: states,
                self.action_gradient: action_gradients}
        self.sess.run(self.opt, feed_dict=feed)

    def save(self, path="checkpoints/autosave_ddpg.ckpt"):
        saver = tf.train.Saver()
        saver.save(self.sess, path)