import torch.nn as nn


class DenseResidualBlock(nn.Module):
    def __init__(self, input_dim, units):
        super(DenseResidualBlock, self).__init__()
        self.dense1 = nn.Linear(input_dim, units)
        self.dense2 = nn.Linear(units, units)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.layer_norm = nn.LayerNorm(units)

        if input_dim != units:
            self.shortcut = nn.Linear(input_dim, units)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)

        x = self.dense1(x)
        x = self.leaky_relu(x)

        x = self.dense2(x)
        x = self.leaky_relu(x)

        x = x + shortcut
        x = self.leaky_relu(x)
        x = self.layer_norm(x)

        return x

class Conv1DResidual(nn.Module):
    def __init__(self, input_channels, filters, kernel_size=3, pool=True):
        super(Conv1DResidual, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, filters, kernel_size, padding='same')
        self.conv2 = nn.Conv1d(filters, filters, kernel_size, padding='same')
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.pool = pool
        self.avg_pool = nn.AvgPool1d(kernel_size=2) if pool else nn.Identity()

        if input_channels != filters:
            self.residual = nn.Conv1d(input_channels, filters, kernel_size=1, padding='same')
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        residual = self.residual(x)

        x = self.conv1(x)
        x = self.leaky_relu(x)

        x = self.conv2(x)
        x = self.leaky_relu(x)

        if self.pool:
            x = self.avg_pool(residual + x)
        else:
            x = x + residual

        return x

class HistoryHandlerBlock(nn.Module):
    def __init__(self, window_size, latent_size):
        super(HistoryHandlerBlock, self).__init__()
        self.conv1 = Conv1DResidual(latent_size, 24, kernel_size=5, pool=True)
        self.conv2 = Conv1DResidual(24, 32, kernel_size=5, pool=True)
        self.conv3 = Conv1DResidual(32, 48, kernel_size=3, pool=True)
        self.conv4 = Conv1DResidual(48, 48, kernel_size=3, pool=False)
        self.conv5 = Conv1DResidual(48, 48, kernel_size=3, pool=False)
        self.flatten = nn.Flatten()
        self.layer_norm = nn.LayerNorm(int(window_size / 8) * 48)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.flatten(x)
        x = self.layer_norm(x)
        return x
