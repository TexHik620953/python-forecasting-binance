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

optimizer = optim.Adam(autoencoder.parameters(), lr=1e-4, weight_decay=0.001)

mult = (torch.log(torch.arange(0,1,1/64)*10+1.1)/2.5).reshape(1,1,-1).to(device)

global_step = 0
for epoch in range(500):
    for batch, _ in generate_window_batches(dataset, batch_size=1000, window_size=64):
        batch = normalize_window_batch(batch)
        batch = torch.from_numpy(batch).float().to(device)

        optimizer.zero_grad()
        encoded, decoded = autoencoder(batch)

        loss = torch.sum(torch.square(decoded - batch) * mult, axis=(1,2))
        diff_loss = torch.sum(torch.square(torch.diff(decoded, axis=2) - torch.diff(batch, axis=2)) * mult[:,:,:-1], axis=(1,2))

        loss = torch.mean(loss) + torch.mean(diff_loss)
        loss.backward()
        optimizer.step()

        encoded = encoded.detach().cpu().numpy()
        loss = float(loss.detach().cpu().numpy())

        tensorboard.add_scalar("loss", loss, global_step)
        tensorboard.add_histogram("encoded", encoded, global_step)

        global_step+=1

        if global_step % 200 == 0:
            plt.plot(batch.detach().cpu().numpy()[0, 2, :], color='black')
            plt.plot(decoded.detach().cpu().numpy()[0, 2, :])
            plt.show()

        if global_step % 100 == 0:
            print(f"[{global_step}]Loss: {loss}")
            autoencoder.save()

