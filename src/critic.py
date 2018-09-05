import tensorflow as tf


class Critic(object):
    def __init__(self, action_space, observation_space, sess):
        """

        :param action_space: list
        :param observation_space: list
        :param sess:
        """
        # action_dim = action_space.shape[0]
        # observation_dim = observation_space.shape[0]

        # gym環境ではないのでリスト型に対応させる
        action_dim = len(action_space)
        observation_dim = len(observation_space)

        self.inputs = tf.placeholder(tf.float32, [None, observation_dim])
        self.actions = tf.placeholder(tf.float32, [None, action_dim])
        fc1 = tf.contrib.layers.fully_connected(self.inputs, 400)
        W1 = tf.Variable(tf.random_normal([400, 300], mean=0.0, stddev=0.05))
        W2 = tf.Variable(tf.random_normal([action_dim, 300],
                                          mean=0.0, stddev=0.05))
        b2 = tf.Variable(tf.zeros([300]))
        fc2 = tf.nn.relu(tf.matmul(fc1, W1) + tf.matmul(self.actions, W2) + b2)
        # fc1 = tf.concat([self.inputs, self.actions], 1)
        # fc2 = tf.contrib.layers.fully_connected(fc1, 64)
        # fc3 = tf.contrib.layers.fully_connected(fc2, 64)
        self.output = tf.contrib.layers.fully_connected(fc2, action_dim,
                                                        activation_fn=None)

        self.targetQs = tf.placeholder(tf.float32, [None, action_dim])
        # self.loss = tf.reduce_mean(tf.square(self.targetQs - self.output))
        self.loss = tf.losses.huber_loss(self.targetQs, self.output)
        self.opt = tf.train.AdamOptimizer(0.001).minimize(self.loss)

        self.action_grads = tf.gradients(self.output, self.actions)

        # self.sess = tf.Session()
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())
        self.network_params = tf.trainable_variables()
        # network_param = self.sess.run(self.network_params)
        # print(network_param)
        # print(len(network_param))

        self.update_network = [self.network_params[i].assign(
            tf.multiply(self.network_params[i], 0.03) + tf.multiply(self.network_params[i + 7], 0.97)) for i in
                               range(1)]

    def train(self, states, actions, rewards, next_qs):
        targetQs = rewards.reshape(20, 1) + 0.99 * next_qs
        feed = {self.inputs: states,
                self.actions: actions,
                self.targetQs: targetQs}
        loss, q, _ = self.sess.run([self.loss, self.output, self.opt],
                                   feed_dict=feed)
        return loss, q

    def get_qs(self, states, actions):
        feed = {self.inputs: states,
                self.actions: actions}
        qs = self.sess.run(self.output, feed_dict=feed)
        return qs

    def get_action_gradients(self, states, actions):
        feed = {self.inputs: states,
                self.actions: actions}
        action_gradients = self.sess.run(self.action_grads, feed_dict=feed)
        return action_gradients

    def update_network_params(self, ):
        self.sess.run(self.update_network)
