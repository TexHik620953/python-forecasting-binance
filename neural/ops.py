import torch.nn as nn
import torch
import math

from torch.nn.modules.module import T


class DenseResidualBlock(nn.Module):
    def __init__(self, input_dim, units, dropout_rate=0.2, use_layer_norm=True):
        super(DenseResidualBlock, self).__init__()
        # Основные слои
        self.dense1 = nn.Linear(input_dim, units)
        self.dense2 = nn.Linear(units, units)

        # Активация и нормализация
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout_rate)

        # Слой нормализации (опционально)
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(units)

        # Shortcut connection
        if input_dim != units:
            self.shortcut = nn.Sequential(
                nn.Linear(input_dim, units),
                nn.LayerNorm(units) if use_layer_norm else nn.Identity()
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)

        # Первый линейный слой
        x = self.dense1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)  # Dropout после активации

        # Второй линейный слой
        x = self.dense2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)  # Dropout после активации

        # Residual connection
        x = x + shortcut

        # Финальная активация и нормализация
        x = self.leaky_relu(x)
        if self.use_layer_norm:
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


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, hidden_dim]

    def forward(self, x):
        # x: [Batch, Time, Features]
        return x + self.pe[:, :x.size(1)]


class MultiHeadAttentionBlock(nn.Module):
    """Один блок Self-Attention с LayerNorm и Residual connection"""

    def __init__(self, hidden_dim=64, num_heads=8, dropout=0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # Используем batch_first=True для удобства
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.dropout0 = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)

    def forward(self, x):
        # x: [Batch, Time, Features]

        # Self-Attention
        attn_out, _ = self.attention(x, x, x)
        attn_out = self.dropout0(attn_out)
        x = self.norm1(x + attn_out)  # Residual + Norm
        # Feed Forward
        ff_out = self.ffn(x)
        ff_out = self.dropout1(ff_out)
        x = self.norm2(x + ff_out)  # Residual + Norm
        return x
    @torch.no_grad()
    def process(self, x):
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        ff_out = self.ffn(x)
        x = self.norm2(x + ff_out)
        return x