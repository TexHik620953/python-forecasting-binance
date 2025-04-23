import os

from neural.ops import DenseResidualBlock, Conv1DResidual, Conv1DUpsampleResidual, MultiHeadAttentionBlock, PositionalEncoding
import torch.nn as nn
import torch
from torch.functional import F

class SimpleClassifier(nn.Module):
    def __init__(self, name, input_dim=33, hidden_dim=64, seq_length=512):
        super().__init__()
        self.name = name

        # CLS-токен как обучаемый параметр (1 дополнительный токен)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))  # [1, 1, D]

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        # Позиционное кодирование
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=seq_length+1)

        # Блоки внимания
        self.attention_blocks = nn.ModuleList([
            MultiHeadAttentionBlock(
                hidden_dim=hidden_dim,
                num_heads=4,
                dropout=0.1
            ) for _ in range(3)
        ])

        self.norm = nn.LayerNorm(hidden_dim)

        # Классификатор
        self.main_dense = DenseResidualBlock(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.output_head = nn.Linear(hidden_dim, 3)

    def forward(self, x):
        x = self.input_proj(x)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)  # [B, 1, D]
        x = torch.cat([cls_tokens, x], dim=1)  # [B, T+1, D]
        # Добавляем позиционное кодирование
        x = self.pos_encoder(x)

        # Применяем несколько блоков внимания
        for attention_block in self.attention_blocks:
            x = attention_block(x)

        cls_output = x[:, 0, :]  # [B, D]
        cls_output = self.norm(cls_output)

        features = self.main_dense(cls_output)
        features = F.leaky_relu(features, negative_slope=0.2)
        features = self.dropout(features)

        return self.output_head(features)
    @torch.no_grad()
    def process(self, x):
        x = self.input_proj(x)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)  # [B, 1, D]
        x = torch.cat([cls_tokens, x], dim=1)  # [B, T+1, D]
        # Добавляем позиционное кодирование
        x = self.pos_encoder(x)

        # Применяем несколько блоков внимания
        for attention_block in self.attention_blocks:
            x = attention_block.process(x)

        cls_output = x[:, 0, :]  # [B, D]
        cls_output = self.norm(cls_output)

        features = self.main_dense(cls_output)
        features = F.leaky_relu(features, negative_slope=0.2)

        return self.output_head(features)

    def save(self):
        os.makedirs(f"./checkpoints", exist_ok=True)
        torch.save({'net': self.state_dict()}, f"./checkpoints/{self.name}.pth")

    def load(self, strict=True):
        dat = torch.load(f"./checkpoints/{self.name}.pth")
        if strict:
            self.load_state_dict(dat['net'])
        else:
            # Копируем только те веса, у которых совпадают размерности
            for name, param in dat['net'].items():
                if name in self.state_dict():
                    if param.size() == self.state_dict()[name].size():
                        self.state_dict()[name].copy_(param)
                    else:
                        print(f"Пропущен {name}: размеры не совпадают ({param.size()} != {self.state_dict()[name].size()})")
