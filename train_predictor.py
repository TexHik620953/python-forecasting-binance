import numpy as np

from dataset.generator import Dataset
from torch.utils.tensorboard import SummaryWriter
from neural.predictor import *
import torch
import torch.optim as optim
import math

tensorboard = SummaryWriter("./tensorboard/predictor")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


dataset = Dataset()
dataset.generate_dataset(dataset.test_data)

predictor = SimpleClassifier("market-transformer").to(device)
predictor.load()

optimizer = optim.Adam(predictor.parameters(), lr=1e-3, weight_decay=2e-5)
optimizer.load_state_dict(torch.load("./checkpoints/optimizer.pth")['opt'])

@torch.no_grad()
def net_test():
    r = []
    for x,y in dataset.generate_dataset(dataset.test_data, batch_size=1000):
        with torch.no_grad():
            y = torch.from_numpy(y).float().to(device)
            x = torch.from_numpy(x).float().to(device)
            logits = predictor(x)
            acc = (logits.detach().argmax(dim=1) == y.argmax(dim=1)).float().mean().item()
            r.append(acc)
    return np.mean(r)

global_step = 0
for epoch in range(2000):
    print("New Epoch")
    for x, y in dataset.generate_dataset(dataset.train_data, batch_size=1000):
        # Train
        y = torch.from_numpy(y).float().to(device)

        x = torch.from_numpy(x).float().to(device)

        optimizer.zero_grad()
        logits = predictor(x)

        loss = torch.mean(torch.sqrt(torch.square(logits - y).sum(dim=1)))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        optimizer.step()

        loss = loss.detach().item()

        tensorboard.add_scalar("train/loss", loss, global_step)
        acc = (logits.detach().argmax(dim=1) == y.argmax(dim=1)).float().mean().item()
        tensorboard.add_scalar("train/accuracy", acc, global_step)

        if global_step % 100 == 0:
            print(f"[{global_step}]Loss: {loss} Accuracy: {math.floor(acc*1000)/10}")
            predictor.save()
            torch.save({'opt': optimizer.state_dict()}, "./checkpoints/optimizer.pth")
            print("Running test")
            test_acc = net_test()
            tensorboard.add_scalar("test/accuracy", test_acc, global_step)


        global_step += 1
