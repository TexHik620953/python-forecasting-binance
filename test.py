import numpy as np
from dataset.generator import Dataset
from neural.predictor import *
import torch
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = Dataset()

predictor = SimpleClassifier("market-transformer").to(device)
predictor.load(strict=False)


def net_test():
    gg = None
    for x, y in dataset.get_test_generator(batch_size=50):
        x = torch.tensor(x, requires_grad=True).float().to(device)

        logits = predictor(x)

        result = logits.square().sum()

        grads = torch.autograd.grad(result, x)
        gs = grads[0].square().sum(dim=0).sum(dim=1).cpu().numpy()
        if gg is None:
            gg = gs
        else:
            gg = gg + gs
    return gg

r = net_test()

plt.plot(r)
plt.show()