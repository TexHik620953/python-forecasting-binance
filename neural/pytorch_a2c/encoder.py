import torch.nn as nn
import torch
from neural.pytorch_a2c.ops import *

class Encoder(nn.Module):
    def __init__(self, window_size, window_latent_size, features_size, internal_features_size):
        super(Encoder, self).__init__()
        self.history_handler = HistoryHandlerBlock(window_size, window_latent_size)
        self.dense1 = DenseResidualBlock(int(window_size / 8) * 48 + features_size, 128)
        self.dense2 = DenseResidualBlock(128, 128)
        self.dense3 = DenseResidualBlock(128, 128)
        self.dense4 = DenseResidualBlock(128, 128)
        self.dense5 = DenseResidualBlock(128, 128)
        self.dense6 = DenseResidualBlock(128, 128)
        self.dropout = nn.Dropout(0.05)
        self.dense7 = DenseResidualBlock(128, internal_features_size)

    def forward(self, x):
        window_input, features = x
        window_features = self.history_handler(window_input)
        common = torch.cat([window_features, features], dim=1)
        common = self.dense1(common)
        common = self.dense2(common)
        common = self.dense3(common)
        common = self.dense4(common)
        common = self.dense5(common)
        common = self.dense6(common)
        common = self.dropout(common)
        encoded_features = self.dense7(common)
        return encoded_features