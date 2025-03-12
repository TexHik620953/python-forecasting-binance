import torch.nn as nn
import torch
from neural.pytorch_a2c.ops import *
import torch.nn.init as init

class Actor(nn.Module):
    def __init__(self, encoder, internal_features_size, actions_amount):
        super(Actor, self).__init__()
        self.encoder = encoder
        self.dense1 = DenseResidualBlock(internal_features_size, 128)
        self.dense2 = DenseResidualBlock(128, 128)
        self.dense3 = DenseResidualBlock(128, 128)
        self.dense4 = DenseResidualBlock(128, 128)
        self.dense5 = DenseResidualBlock(128, 128)
        self.dropout = nn.Dropout(0.05)
        self.dense6 = nn.Linear(128, actions_amount)
        self.softmax = nn.Softmax(dim=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.5, mean=0)
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, x):
        x = self.encoder(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.dense5(x)
        x = self.dropout(x)
        x = self.dense6(x)
        x = self.softmax(x)
        return x