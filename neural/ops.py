import torch.nn as nn
import torch
import math

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


class Conv1DUpsampleResidual(nn.Module):
    def __init__(self, input_channels, filters, kernel_size=3, scale_factor=2):
        super(Conv1DUpsampleResidual, self).__init__()

        self.conv1 = nn.Conv1d(input_channels, filters, kernel_size, padding='same')
        self.conv2 = nn.Conv1d(filters, filters, kernel_size, padding='same')
        self.leaky_relu = nn.LeakyReLU(0.2)

        # Апскейлинг
        self.upsample = nn.ConvTranspose1d(filters, filters, kernel_size=scale_factor, stride=scale_factor)

        # Остаточная связь
        if input_channels != filters:
            self.residual = nn.Sequential(
                nn.Conv1d(input_channels, filters, kernel_size=1, padding='same'),
                nn.Upsample(scale_factor=scale_factor, mode='nearest')
            )
        else:
            self.residual = nn.Upsample(scale_factor=scale_factor, mode='nearest')

    def forward(self, x):
        residual = self.residual(x)

        # Основной путь
        x = self.conv1(x)
        x = self.leaky_relu(x)

        x = self.conv2(x)
        x = self.leaky_relu(x)

        # Апскейлинг
        x = self.upsample(x)

        # Остаточная связь
        return x + residual

class Conv1DResidual(nn.Module):
    def __init__(self, input_channels, filters, kernel_size=3, pool=True):
        super(Conv1DResidual, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, filters, kernel_size, padding='same')
        self.bn1 = nn.BatchNorm1d(filters)  # Нормализация после conv1
        self.conv2 = nn.Conv1d(filters, filters, kernel_size, padding='same')
        self.bn2 = nn.BatchNorm1d(filters)  # Нормализация после conv2
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.pool = pool
        self.avg_pool = nn.MaxPool1d(kernel_size=2) if pool else nn.Identity()

        if input_channels != filters:
            self.residual = nn.Conv1d(input_channels, filters, kernel_size=1, padding='same')
            self.residual_bn = nn.BatchNorm1d(filters)  # Нормализация для residual (если меняется размерность)
        else:
            self.residual = nn.Identity()
            self.residual_bn = nn.Identity()  # Identity, если размерности совпадают

    def forward(self, x):
        residual = self.residual(x)
        residual = self.residual_bn(residual)  # Применяем BatchNorm к residual

        x = self.conv1(x)
        x = self.bn1(x)  # BatchNorm перед активацией
        x = self.leaky_relu(x)

        x = self.conv2(x)
        x = self.bn2(x)  # BatchNorm перед активацией
        x = self.leaky_relu(x)

        if self.pool:
            x = self.avg_pool(residual + x)
        else:
            x = x + residual

        return x



class MultiHeadAttentionBlock(nn.Module):
    """Один блок Self-Attention с LayerNorm и Residual connection"""

    def __init__(self, hidden_dim=64, num_heads=8, dropout=0.05):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [Batch, Time, Features]
        x = x.transpose(0, 1)  # MultiheadAttention ожидает [Time, Batch, Features]

        # Self-Attention
        attn_out, _ = self.attention(x, x, x)
        attn_out = self.dropout(attn_out)
        x = self.norm1(x + attn_out)  # Residual + Norm

        # Feed Forward
        ff_out = self.ffn(x)
        x = self.norm2(x + ff_out)  # Residual + Norm

        return x.transpose(0, 1)  # Возвращаем к [Batch, Time, Features]


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2) * (-math.log(10000.0) / hidden_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]