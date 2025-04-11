import numpy as np
from fontTools.misc.cython import returns


class Dataset:
    def __init__(self):
        self.data = np.load("./dataset/data.npy").swapaxes(1, 0)
        self.window_size = 512
        self.next_window_size = 60

        valudation_split = 0.1
        test_split = 0.1
        datalen = self.data.shape[0]

        self.validation_data = self.data[int(datalen*(1-valudation_split)):]
        self.test_data = self.data[int(datalen * (1 - valudation_split - test_split)):int(datalen*(1-valudation_split))]
        self.train_data = self.data[:int(datalen * (1 - valudation_split - test_split))]

    def normalize_window(self, window):
        std = window.std(axis=0)
        mean = window.mean(axis=0)
        return (window - mean) / (std + 1e-6)

    def generate_dataset(self, dataset, batch_size=500):
        # OPEN HIGH LOW CLOSE VOLUME QUOTE_VOLUME NUMBER_OF TRADES TAKER_VOLUME TAKER_QUOTE_VOLUME
        x = []
        y = []
        for i in range(dataset.shape[0] - self.window_size):
            _x = dataset[i:i+self.window_size].reshape(self.window_size,-1)
            x.append(self.normalize_window(_x))

            current_closes = dataset[i+self.window_size-1][1, 3] # Close

            next_window = dataset[i+self.window_size: i+self.window_size+self.next_window_size]
            next_window_low = np.min(next_window[:,1, 2], axis=0)
            next_window_high = np.max(next_window[:,1, 1], axis=0)

            max_raise = (next_window_high - current_closes) / current_closes
            max_drop = (current_closes - next_window_low) / current_closes

            _y = np.array([max_drop, max_raise])
            _y = _y / (np.sum(_y) + 1e-6)

            y.append(_y)

            if len(x) == batch_size:
                yield np.array(x), np.array(y)
                x = []
                y = []
        if len(x) == batch_size:
            yield np.array(x), np.array(y)