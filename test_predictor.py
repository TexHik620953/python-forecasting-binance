from dataset.dataset_raw import load_pair, DatasetGenerator, normalize_window_batch
from torch.utils.tensorboard import SummaryWriter
from neural.autoencoder import Encoder
from neural.predictor import *
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import math

tensorboard = SummaryWriter("./tensorboard/predictor")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = DatasetGenerator(load_pair(), batch_size=500, window_size=64, next_detection=4)

x,y = dataset.validation()


predictor = SimpleClassifier("market-transformer").to(device)
predictor.load()

v = 0
c = 0
for i in range(len(x)):
    _x = torch.from_numpy(normalize_window_batch(x[i])).float().to(device)
    _y = torch.from_numpy(y[i]).long().to(device)

    p = predictor(_x).detach().argmax(dim=1)

    v += (p == _y).float().mean().item()
    c += 1

v = v / c

print(f"Acc: {v*100}%")