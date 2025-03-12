import os
from random import sample
import numpy as np
from collections import deque


class ReplayBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=self.max_size)

    def add_replay(self, s, a, r, s_new):
        self.buffer.append((s, a, r, s_new))

    def __len__(self):
        return len(self.buffer)

    def pop_minibatch(self, minibatch_size):
        batch = [self.buffer.popleft() for i in range(minibatch_size)]
        s, a, r, s_new = list(map(list, zip(*batch)))
        return s, a, r, s_new

    def sample_random_minibatch(self, minibatch_size):
        batch = sample(self.buffer, minibatch_size)
        s, a, r, s_new = list(map(list, zip(*batch)))
        return s, a, r, s_new

    def save(self):
        os.makedirs("./checkpoints", exist_ok=True)
        np.save("./checkpoints/replay_buffer.npy", {"data":list(self.buffer)}, allow_pickle=True)
        print("Saved trajectory buffer")

    def load(self):
        try:
            temp = np.load("./checkpoints/replay_buffer.npy", allow_pickle=True)
            temp = list(temp.item()["data"])
            self.buffer = deque(temp, maxlen=self.max_size)
        except:
            print("Failed to load trajectory buffer")
