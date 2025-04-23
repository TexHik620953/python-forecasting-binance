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

predictor = SimpleClassifier("market-transformer").to(device)
predictor.load(strict=True)

optimizer = optim.Adam(predictor.parameters(), lr=5e-4, weight_decay=2e-5)
optimizer.load_state_dict(torch.load("checkpoints/optimizer.pth")['opt'])

@torch.no_grad()
def net_test():
    _acc = []
    for x, y in dataset.get_test_generator(batch_size=100):
        with torch.no_grad():
            x = torch.from_numpy(x).float().to(device)
            y = torch.from_numpy(y).float().to(device)

            pr_logits = predictor.process(x)

            accuracy = (pr_logits.detach().argmax(dim=1) == y.argmax(dim=1)).float().cpu().mean().item()
            _acc.append(accuracy)
    return np.mean(_acc)


global_step = 1
for epoch in range(2000):
    print("New Epoch")
    for x, y in dataset.get_train_generator(batch_size=100):
        # Train
        x = torch.from_numpy(x).float().to(device)
        y = torch.from_numpy(y).float().to(device)

        optimizer.zero_grad()
        pr_logits = predictor(x)

        loss = torch.mean(torch.square(pr_logits - y).sum(dim=1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        optimizer.step()

        loss = loss.detach().item()

        tensorboard.add_scalar("train/loss", loss, global_step)
        accuracy = (pr_logits.detach().argmax(dim=1) == y.argmax(dim=1)).float().cpu().mean().item()

        tensorboard.add_scalar("train/accuracy", accuracy, global_step)

        if global_step % 10000 == 0:
            print(f"[{global_step}]Loss: {loss} Accuracy: {math.floor(accuracy*1000)/10}")
            ok = False
            while not ok:
                try:
                    predictor.save()
                    torch.save({'opt': optimizer.state_dict()}, "checkpoints/optimizer.pth")
                    ok = True
                    break
                except:
                    ok = False

        if global_step % 2000 == 0:
            print("Running test")
            test_acc = net_test()
            tensorboard.add_scalar("test/accuracy", test_acc, global_step)

        global_step += 1
