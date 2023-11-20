import numpy as np
import torch
from sklearn.datasets import make_blobs

from dl_13_pytorch_basics2 import get_device, train, vis_losses_accs, MLP
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def get_grid_data(xlim, ylim):
    x1 = np.linspace(xlim[0], xlim[1], 100)
    x2 = np.linspace(ylim[0], ylim[1], 100)
    X1, X2 = np.meshgrid(x1, x2)
    X_db = np.hstack([X1.reshape(-1, 1), X2.reshape(-1, 1)])
    return X_db


def get_dataset(N_SAMPLES=100, BATCH_SIZE=8):
    HALF_N_SAMPLES = int(N_SAMPLES / 2)

    # create data
    centers1 = [(-1, 1), (-1, -1)]
    centers2 = [(1, -1), (1, 1)]
    X1, y1 = make_blobs(n_samples=HALF_N_SAMPLES, centers=centers1, n_features=2, cluster_std=0.3)
    X2, y2 = make_blobs(n_samples=HALF_N_SAMPLES, centers=centers2, n_features=2, cluster_std=0.3)

    X = np.vstack([X1, X2])
    y = np.vstack([y1, y2]).flatten()

    # grid
    xylim, fig, ax = visualize_dataset(X, y)
    X_db = get_grid_data(*xylim)
    X_db = TensorDataset(torch.FloatTensor(X_db))

    # dataloader
    dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
    return dataloader, dataset, X_db, X, y


def visualize_dataset(X, y):
    fig, ax = plt.subplots(figsize=(10, 10))
    X_pos, X_neg = X[y == 1], X[y == 0]
    ax.scatter(X_pos[:, 0], X_pos[:, 1], color='blue')
    ax.scatter(X_neg[:, 0], X_neg[:, 1], color='red')
    ax.tick_params(labelsize=15)
    # ax.scatter(X[:, 0], X[:, 1], c=y)
    # fig.tight_layout()
    # plt.show()

    xylim = ax.get_xlim(), ax.get_ylim()
    return xylim, fig, ax


def vis_meshgrid(X_db, pred, X, y):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(X_db[:, 0], X_db[:, 1], c=pred, cmap='bwr', alpha=0.1)

    X_pos, X_neg = X[y == 0], X[y == 1]
    ax.scatter(X_pos[:, 0], X_pos[:, 1], color='blue')
    ax.scatter(X_neg[:, 0], X_neg[:, 1], color='red')
    plt.show()


if __name__ == '__main__':
    np.random.seed(500)
    N_SAMPLES = 300
    BATCH_SIZE = 8
    EPOCHS = 1000
    LR = 0.01
    n_features = 2
    DEVICE = get_device()
    dataloader, dataset, X_db, X, y = get_dataset(300)

    model = MLP(n_features=n_features).to(DEVICE)
    loss_function = nn.BCELoss()
    optimizer = SGD(model.parameters(), lr=LR)

    losses, accs = list(), list()

    for epoch in range(EPOCHS):
        epoch_loss, epoch_acc = train(dataloader, N_SAMPLES, model,
                                      loss_function, optimizer, DEVICE)

        losses.append(epoch_loss)
        accs.append(epoch_acc)

        if epoch % 100 == 0:
            print(f"EPOCH: {epoch + 1}", end="\t")
            print(f"Accuracy: {epoch_acc}", end="\t")
            print(f"Loss: {epoch_loss}")

    X_db = X_db.tensors[0].to(DEVICE)
    pred = model.forward(X_db)

    X_db = X_db.to("cpu").detach().numpy()
    pred = pred.to("cpu").detach()
    pred = (pred > 0.5).type(torch.float).numpy()

    # fig1
    vis_losses_accs(losses, accs)
    # fig2
    vis_meshgrid(X_db, pred, X, y)





