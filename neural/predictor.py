import os

from neural.ops import DenseResidualBlock, Conv1DResidual, Conv1DUpsampleResidual, MultiHeadAttentionBlock, PositionalEncoding
import torch.nn as nn
import torch
from torch.functional import F

class SimpleClassifier(nn.Module):
    def __init__(self, name, input_dim=36, hidden_dim=96, num_heads=4, num_layers=3):
        super().__init__()
        self.name = name

        # Преобразование входных данных
        self.input_proj = nn.Sequential(
            Conv1DResidual(input_dim, hidden_dim, kernel_size=5, pool=True),
            Conv1DResidual(hidden_dim, hidden_dim, kernel_size=5, pool=True),
            Conv1DResidual(hidden_dim, hidden_dim, kernel_size=5, pool=True),
            Conv1DResidual(hidden_dim, hidden_dim, kernel_size=5, pool=True),
            Conv1DResidual(hidden_dim, hidden_dim, kernel_size=5, pool=True),
            Conv1DResidual(hidden_dim, hidden_dim, kernel_size=5, pool=True),
        )
        #self.pos_encoder = PositionalEncoding(input_dim)

        #self.attention_blocks = nn.ModuleList([MultiHeadAttentionBlock(hidden_dim, num_heads) for _ in range(num_layers)])

        # Классификатор
        self.main_net = nn.ModuleList([
            nn.Linear(768, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2),
        ])

        self.output_proj = nn.Linear(128, 2)

    def forward(self, x):
        # x: [Batch, Features, Time]
        x = x.swapaxes(1, 2)
        x = self.input_proj(x)  # Проекция в скрытое пространство -> [B, Time, Features]
        x = x.reshape(x.shape[0], -1)

        #x = self.pos_encoder(x)
        #for block in self.attention_blocks:
        #    x = block(x)
        '''
        x = torch.cat([
            x.mean(dim=1), # MEAN
            x.max(dim=1)[0], # MAX
            x[:,0, :] # CLS
            ], dim=1)
        '''
        for block in self.main_net:
            x = F.leaky_relu(block(x), negative_slope=0.1)

        x = self.output_proj(x)
        return x

    def save(self):
        os.makedirs(f"./checkpoints", exist_ok=True)
        torch.save({'net': self.state_dict()}, f"./checkpoints/{self.name}.pth")

    def load(self):
        dat = torch.load(f"./checkpoints/{self.name}.pth")
        self.load_state_dict(dat['net'])