import numpy as np
from collections import deque


class Memory():
    def __init__(self, max_size=10000):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        # np.random.choice リストからsize個をランダムチョイス,
        # np.arange xまでの連番の配列
        idx = np.random.choice(np.arange(len(self.buffer)),
                               size=batch_size,
                               replace=False)
        batch = [self.buffer[i] for i in idx]
        states = np.array([each[0] for each in batch])
        actions = np.array([each[1] for each in batch])
        rewards = np.array([each[2] for each in batch])
        next_states = np.array([each[3] for each in batch])
        return states, actions, rewards, next_states