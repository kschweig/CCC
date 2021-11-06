import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 8, 4, stride=2)
        self.conv2 = nn.Conv2d(8, 16, 3, stride=2)
        self.conv3 = nn.Conv2d(16, 24, 3, stride=1)
        self.conv4 = nn.Conv2d(24, 32, 3, stride=1)
        self.drop = nn.Dropout(p=0.1)
        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(24)
        self.bn4 = nn.BatchNorm2d(32)
        self.act = nn.ReLU()
        self.hidden = nn.Linear(32 * 2 * 16, 256)
        self.out = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.FloatTensor(x)

        if len(x.shape) == 3:
            x.unsqueeze(0)

        x = self.drop(self.act(self.bn1(self.conv1(x))))
        x = self.drop(self.act(self.bn2(self.conv2(x))))
        x = self.drop(self.act(self.bn3(self.conv3(x))))
        x = self.drop(self.act(self.bn4(self.conv4(x))))
        x = torch.flatten(x, start_dim=1)
        x = self.act(self.hidden(x))
        return torch.sigmoid(self.out(x)).squeeze(1)


class MyDS(torch.utils.data.Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        return torch.FloatTensor(self.x[item].reshape(1, 28, -1)), self.y[item]


def _forward(network: nn.Module, data: DataLoader, metric: callable):
    device = next(network.parameters()).device

    for x, y in data:
        x, y = x.to(device), y.to(device=device, dtype=torch.float32)
        outputs = network(x)
        res = metric(outputs, y)
        yield res


@torch.no_grad()
def evaluate(network: nn.Module, data: DataLoader, metric: callable) -> list:
    network.eval()

    results = _forward(network, data, metric)
    return np.mean([res.item() for res in results])


@torch.enable_grad()
def update(network: nn.Module, data: DataLoader, loss: nn.Module,
           opt: optim.Optimizer) -> list:
    network.train()

    errs = []
    for err in _forward(network, data, loss):
        errs.append(err.item())

        opt.zero_grad()
        err.backward()
        opt.step()

    return errs

def accuracy(y_hat, y):
    y_hat = y_hat.detach().numpy()
    y = y.detach().numpy()

    y_hat = [1. if y_ > 0.5 else 0. for y_ in y_hat]
    return accuracy_score(y, y_hat)


images = []
targets = []

with open(os.path.join("data", f"train.csv"), "r") as f:
    for l, line in enumerate(f):
        if l == 0:
            N = int(line)
            print(N)
        elif l <= N:
            images.append([int(s) for s in line.split(",")])
        else:
            targets.append(int(line))

images = np.asarray(images)
print(N, images.shape, len(targets))

norm = np.max(images)

images = images / norm

xtrain, xtest, ytrain, ytest = train_test_split(images, targets, test_size=0.05, shuffle=True)


lr = 1e-3
batch_size = 64
epochs = 30


model = CNN()
optimizer = optim.Adam(params=model.parameters(), lr=lr, weight_decay=0)
loss = nn.BCELoss()

train_loader = DataLoader(MyDS(xtrain, ytrain), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(MyDS(xtest, ytest), batch_size=len(xtest), shuffle=False)

for ep in range(epochs):
    print("loss:", np.mean(update(network=model, data=train_loader, loss=loss, opt=optimizer)))
    print("acc:", evaluate(network=model, data=test_loader, metric=accuracy))

for c in range(1, 3):

    model.eval()

    preds = []
    images = []

    with open(os.path.join("data", f"level_2_{c}.csv"), "r") as f:

        for l, line in enumerate(f):
            if l == 0:
                N = int(line)
                print(N)
            else:
                images.append([int(s) for s in line.split(",")])

    images = np.asarray(images)
    images = images / norm

    for image in images:
        pred = model.forward(torch.FloatTensor(image.reshape(1,1,28, -1)))
        if pred > 0.5:
            preds.append(1)
        else:
            preds.append(0)


    with open(os.path.join("results", f"level_2_{c}.csv"), "w") as f:
        for r in preds:
            f.write(f"{int(r)}\n")
