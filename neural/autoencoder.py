import os

from neural.ops import DenseResidualBlock, Conv1DResidual, Conv1DUpsampleResidual
import torch.nn as nn
import torch

class Encoder(nn.Module):
    def __init__(self,name):
        super(Encoder, self).__init__()
        self.name = name

        self.encoder = nn.Sequential(
            Conv1DResidual(5, 8, kernel_size=3, pool=False),
            Conv1DResidual(8, 8, kernel_size=3, pool=True),

            Conv1DResidual(8, 16, kernel_size=3, pool=False),
            Conv1DResidual(16, 16, kernel_size=3, pool=True),

            Conv1DResidual(16, 32, kernel_size=3, pool=False),
            Conv1DResidual(32, 32, kernel_size=3, pool=True),

            Conv1DResidual(32, 64, kernel_size=3, pool=False),
            Conv1DResidual(64, 64, kernel_size=3, pool=True),

            Conv1DResidual(64, 128, kernel_size=3, pool=False),
            Conv1DResidual(128, 128, kernel_size=3, pool=True),

            Conv1DResidual(128, 256, kernel_size=3, pool=False),
            Conv1DResidual(256, 256, kernel_size=3, pool=True),

            nn.Flatten(),

            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
        )
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in')
            m.bias.data.fill_(0.01)

    def forward(self, x):
        x = self.encoder(x)
        return x

    def save(self):
        os.makedirs(f"./checkpoints", exist_ok=True)
        torch.save({'net': self.state_dict()}, f"./checkpoints/{self.name}.pth")

    def load(self):
        dat = torch.load(f"./checkpoints/{self.name}.pth")
        self.load_state_dict(dat['net'])


class Decoder(nn.Module):
    def __init__(self,name):
        super(Decoder, self).__init__()
        self.name = name

        self.decoder = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),

            nn.Unflatten(1, (256, 1)),

            Conv1DUpsampleResidual(256, 256, kernel_size=3, scale_factor=1),
            Conv1DUpsampleResidual(256, 128, kernel_size=3, scale_factor=2),

            Conv1DUpsampleResidual(128, 128, kernel_size=3, scale_factor=1),
            Conv1DUpsampleResidual(128, 64, kernel_size=3, scale_factor=2),

            Conv1DUpsampleResidual(64, 64, kernel_size=3, scale_factor=1),
            Conv1DUpsampleResidual(64, 32, kernel_size=3, scale_factor=2),

            Conv1DUpsampleResidual(32, 16, kernel_size=3, scale_factor=1),
            Conv1DUpsampleResidual(16, 16, kernel_size=3, scale_factor=2),

            Conv1DUpsampleResidual(16, 8, kernel_size=3, scale_factor=1),
            Conv1DUpsampleResidual(8, 8, kernel_size=3, scale_factor=2),

            Conv1DUpsampleResidual(8, 5, kernel_size=3, scale_factor=1),
            Conv1DUpsampleResidual(5, 5, kernel_size=3, scale_factor=2),
        )

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in')
            m.bias.data.fill_(0.01)

    def forward(self, x):
        x = self.decoder(x)
        return x

    def save(self):
        os.makedirs(f"./checkpoints/", exist_ok=True)
        torch.save({'net': self.state_dict()}, f"./checkpoints/{self.name}.pth")

    def load(self):
        dat = torch.load(f"./checkpoints/{self.name}.pth")
        self.load_state_dict(dat['net'])



class AutoEncoder(nn.Module):
    def __init__(self, name):
        super(AutoEncoder, self).__init__()
        self.name = name

        self.encoder = Encoder(name + "encoder")
        self.decoder = Decoder(name + "decoder")

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return encoded, decoded

    def save(self):
        self.encoder.save()
        self.decoder.save()

    def load(self):
        self.encoder.load()
        self.decoder.load()