import os

import numpy as np
import random
from collections import deque


class Dataset:
    def __init__(self):
        data_parts = []
        for file in os.listdir("./dataset"):
            if file.endswith(".npy"):
                data_parts.append(os.path.join("./dataset", file))
        data_parts = sorted(data_parts)
        self.validation_part = [data_parts[0]]
        self.test_part = data_parts[1]
        self.data_parts = data_parts[2:]
        print()

    def __generate_data(self, filename):
        f = np.load(filename, allow_pickle=True)
        x = f.item()['x']
        y = f.item()['y']
        for i in range(len(x)):
            _x, _y = x[i], y[i]

            if np.isnan(_x).any() or np.isnan(_y).any():
                continue

            l = None
            if abs(_y) < 0.2/100: # 0.2%
                l = [0, 1, 0]
            else:
                if _y > 0:
                    l = [0, 0, 1]
                else:
                    l = [1, 0, 0]

            yield _x, l

    def get_train_generator(self, batch_size=500):
        X, Y = [], []
        for f in self.data_parts:
            for x, y in self.__generate_data(f):

                X.append(x)
                Y.append(y)
                if len(X) >= batch_size:
                    yield np.array(X), np.array(Y)
                    X, Y = [], []
        if len(X) > 0:
            yield np.array(X), np.array(Y)

    def get_test_generator(self, batch_size=500):
        X, Y = [], []
        for x, y in self.__generate_data(self.test_part):
            X.append(x)
            Y.append(y)
            if len(X) >= batch_size:
                yield np.array(X), np.array(Y)
                X, Y = [], []
        if len(X) > 0:
            yield np.array(X), np.array(Y)