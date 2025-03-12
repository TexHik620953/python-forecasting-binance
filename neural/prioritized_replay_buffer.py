import random
import numpy as np
from collections import deque
import os

class PrioritizedReplayBuffer:
    def __init__(self, max_size, alpha=0.6, beta=0.4):
        self.max_size = max_size
        self.alpha = alpha  # Параметр приоритета (0 <= alpha <= 1)
        self.beta = beta    # Параметр коррекции смещения (0 <= beta <= 1)
        self.buffer = deque(maxlen=max_size)
        self.priorities = {}
        self.pos = 0
        self.size = 0

    def add_replay(self, s, a, r, s_new):
        if len(self.buffer) < self.max_size:
            self.buffer.append((s, a, r, s_new))
        else:
            self.buffer[self.pos] = (s, a, r, s_new)

        for b_id in self.priorities.keys():
            self.priorities[b_id][self.pos] = 1
        self.pos = (self.pos + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def __len__(self):
        return len(self.buffer)

    def sample_random_minibatch(self, brain_id, minibatch_size):
        if self.size == 0:
            return []
        if brain_id not in self.priorities.keys():
            self.priorities[brain_id] = np.ones((self.max_size,), dtype=np.float32)

        # Вычисляем вероятности выборки на основе приоритетов
        priorities = self.priorities[brain_id][:self.size] ** self.alpha
        probs = priorities / priorities.sum()

        # Выбираем индексы с учетом вероятностей
        indices = np.random.choice(self.size, minibatch_size, p=probs)
        batch = [self.buffer[idx] for idx in indices]

        # Вычисляем веса для коррекции смещения
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        s, a, r, s_new = list(map(list, zip(*batch)))
        return (s, a, r, s_new), indices, weights

    def update_priorities(self, brain_id, indices, priorities, beta = 0.95):
        if brain_id not in self.priorities.keys():
            self.priorities[brain_id] = beta * self.priorities[brain_id] + (1 - beta) * np.ones((self.max_size,), dtype=np.float32)

        for idx, priority in zip(indices, priorities):
            self.priorities[brain_id][idx] = priority

    def save(self):
        #Save position, size and step
        os.makedirs("./checkpoints/", exist_ok=True)
        np.save("./checkpoints/replay_buffer.npy",
                {
                    "data": list(self.buffer),
                    "priorities": self.priorities,
                    "size": self.size,
                    "pos": self.pos,
                    "max_size": self.max_size,
                }, allow_pickle=True)

    def load(self):
        try:
            data = np.load("./checkpoints/replay_buffer.npy", allow_pickle=True)
            temp = data.item()
            self.size = temp["size"]
            self.pos = temp["pos"]
            self.max_size = temp["max_size"]
            self.buffer = deque(temp["data"], maxlen=self.max_size)
            self.priorities = temp["priorities"]
        except Exception as e:
            print(f"Failed to load trajectory buffer: {e}")

