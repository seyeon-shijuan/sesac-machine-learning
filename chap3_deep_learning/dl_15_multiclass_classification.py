from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.datasets import make_blobs

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm


@dataclass
class Constants:
    N_SAMPLES: int
    BATCH_SIZE: int
    EPOCHS: int
    LR: float
    n_features: int
    DEVICE: torch.device
    PATH: str
    SEED: int


def get_device():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"curr device = {DEVICE}")
    return DEVICE


def get_grid_data(X, y):
    # xylim
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='tab10')
    ax.tick_params(labelsize=15)
    plt.show()
    x_lim, y_lim = ax.get_xlim(), ax.get_ylim()
    plt.close()

    # grid
    x1 = np.linspace(x_lim[0], x_lim[1], 100)
    x2 = np.linspace(y_lim[0], y_lim[1], 100)
    X1, X2 = np.meshgrid(x1, x2)
    X_db = np.hstack([X1.reshape(-1, 1), X2.reshape(-1, 1)])
    return X_db

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.fc1 = nn.Linear(in_features=10, out_features=16)
        self.fc1_act = nn.ReLU()
        self.fc2 = nn.Linear(in_features=16, out_features=32)
        self.fc2_act = nn.ReLU()
        self.fc3 = nn.Linear(in_features=32, out_features=10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc1_act(x)
        x = self.fc2(x)
        x = self.fc2_act(x)
        x = self.fc3(x)
        return x


class MultiClassClassifier(nn.Module):
    def __init__(self):
        super(MultiClassClassifier, self).__init__()
        self.fc1 = nn.Linear(in_features=2, out_features=4)
        self.fc1_act = nn.ReLU()
        self.fc2 = nn.Linear(in_features=4, out_features=8)
        self.fc2_act = nn.ReLU()
        self.fc3 = nn.Linear(in_features=8, out_features=4)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc1_act(x)
        x = self.fc2(x)
        x = self.fc2_act(x)
        x = self.fc3(x)
        return x


def get_data(c):
    # create data
    HALF_SAMPLES = int(c.N_SAMPLES / 2)
    centers1 = [(-10, -4), (-7, -8)]
    centers2 = [(-6, -3), (-2, 4)]
    X1, y1 = make_blobs(n_samples=HALF_SAMPLES, centers=centers1, n_features=2, cluster_std=0.5, random_state=c.SEED, shuffle=True)
    X2, y2 = make_blobs(n_samples=HALF_SAMPLES, centers=centers2, n_features=2, cluster_std=0.5, random_state=c.SEED, shuffle=True)
    y2 += 2
    X = np.vstack([X1, X2])
    y = np.vstack([y1, y2]).flatten()

    # grid
    X_db = get_grid_data(X, y)
    X_db = TensorDataset(torch.FloatTensor(X_db))

    # dataloader
    dataset = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
    dataloader = DataLoader(dataset, batch_size=c.BATCH_SIZE, shuffle=True)

    return dataloader, X_db, X, y


def train(dataloader, N_SAMPLES, model, loss_function, optimizer, DEVICE):
    epoch_loss, n_corrects = 0., 0

    for X, y in tqdm(dataloader):
        X, y = X.to(DEVICE), y.to(DEVICE)

        pred = model.forward(X)

        loss = loss_function(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * len(X)
        pred_cls = torch.argmax(pred, dim=1)
        n_corrects += (pred_cls == y).sum().item()

    epoch_loss /= N_SAMPLES
    epoch_accr = n_corrects / N_SAMPLES

    return epoch_loss, epoch_accr


def vis_losses_accs(losses, accs):
    fig, axes = plt.subplots(2, 1, figsize=(14, 5))
    axes[0].plot(losses)
    axes[1].plot(accs)

    axes[1].set_xlabel("Epoch", fontsize=15)
    axes[0].set_ylabel("Loss", fontsize=15)
    axes[1].set_ylabel("Accuracy", fontsize=15)
    axes[0].tick_params(labelsize=10)
    axes[1].tick_params(labelsize=10)
    fig.suptitle("Multiclass 3-layer Model Metrics by Epoch", fontsize=16)


def vis_meshgrid(X_db, pred, X, y):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr')
    ax.scatter(X_db[:, 0], X_db[:, 1], c=pred, cmap='bwr', alpha=0.1)
    plt.show()


def run_multiclass_classifier(c):
    # Data
    dataloader, X_db, X, y = get_data(constants)

    # Model
    model = MultiClassClassifier().to(c.DEVICE)
    loss_function = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=c.LR)

    # Training
    losses, accs = list(), list()

    for epoch in range(c.EPOCHS):
        epoch_loss, epoch_acc = train(dataloader, c.N_SAMPLES, model, loss_function,
                                      optimizer, c.DEVICE)

        losses.append(epoch_loss)
        accs.append(epoch_acc)

        if epoch % 10 == 0:
            print(f"\n EPOCH: {epoch}", end="\t")
            print(f"Accuracy: {epoch_acc}", end="\t")
            print(f"Loss: {epoch_loss}")


    # Predict
    X_db = X_db.tensors[0].to(c.DEVICE)
    # 1번 방법(faster) torch.no_grad() autograd engine 비활성화
    with torch.no_grad():
        pred = model(X_db).to("cpu")

    X_db = X_db.to("cpu").detach().numpy()
    pred_cls = torch.argmax(pred, dim=1)

    # Visualization
    vis_losses_accs(losses, accs)
    vis_meshgrid(X_db, pred_cls, X, y)


if __name__ == '__main__':
    constants = Constants(
        N_SAMPLES=100,
        BATCH_SIZE=8,
        EPOCHS=100,
        LR=0.01,
        n_features=2,
        DEVICE=get_device(),
        PATH="model/multicls_params.pt",
        SEED=80
    )
    run_multiclass_classifier(constants)

