from dataset.dataset_raw import load_pair, generate_window_batches, normalize_window_batch
from torch.utils.tensorboard import SummaryWriter
from neural.autoencoder import AutoEncoder
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

tensorboard = SummaryWriter("./tensorboard/autoencoder")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = load_pair()

autoencoder = AutoEncoder("autoencoder").to(device)
autoencoder.load()

optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3, weight_decay=0.001)

global_step = 0
for epoch in range(500):
    for batch, _ in generate_window_batches(dataset, batch_size=1000):
        batch = normalize_window_batch(batch)
        batch = torch.from_numpy(batch).float().to(device)

        optimizer.zero_grad()
        encoded, decoded = autoencoder(batch)

        decoded = decoded.detach().cpu().numpy()
        encoded = encoded.detach().cpu().numpy()

        plt.plot(batch.detach().cpu().numpy()[0,2,:], color='black')
        plt.plot(decoded.detach().cpu().numpy()[0,2,:])
        plt.show()