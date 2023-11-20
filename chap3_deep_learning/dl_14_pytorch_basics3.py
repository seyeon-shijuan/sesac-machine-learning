import numpy as np
import torch
from sklearn.datasets import make_blobs

from dl_13_pytorch_basics2 import get_device, train, vis_losses_accs, MLP
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataclasses import dataclass


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

def get_grid_data(xlim, ylim):
    x1 = np.linspace(xlim[0], xlim[1], 100)
    x2 = np.linspace(ylim[0], ylim[1], 100)
    X1, X2 = np.meshgrid(x1, x2)
    X_db = np.hstack([X1.reshape(-1, 1), X2.reshape(-1, 1)])
    return X_db


def get_dataset(N_SAMPLES=100, BATCH_SIZE=8, SEED=0):
    HALF_N_SAMPLES = int(N_SAMPLES / 2)

    # create data
    centers1 = [(-1, -1), (-1, 1)]
    centers2 = [(1, 1), (1, -1)]
    X1, y1 = make_blobs(n_samples=HALF_N_SAMPLES, centers=centers1, n_features=2, cluster_std=0.3, random_state=SEED)
    X2, y2 = make_blobs(n_samples=HALF_N_SAMPLES, centers=centers2, n_features=2, cluster_std=0.3, random_state=SEED)

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
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr')
    ax.tick_params(labelsize=15)
    # plt.show()
    xylim = ax.get_xlim(), ax.get_ylim()
    return xylim, fig, ax


def vis_meshgrid(X_db, pred, X, y):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr')
    ax.scatter(X_db[:, 0], X_db[:, 1], c=pred, cmap='bwr', alpha=0.1)
    plt.show()


def xor_with_mlp(c):
    # Data
    dataloader, dataset, X_db, X, y = get_dataset(N_SAMPLES=c.N_SAMPLES, BATCH_SIZE=c.BATCH_SIZE, SEED=c.SEED)

    # Model
    model = MLP(n_features=c.n_features, n_hdn_nrns=3).to(c.DEVICE)
    loss_function = nn.BCELoss()
    optimizer = SGD(model.parameters(), lr=c.LR)

    # Training
    losses, accs = list(), list()

    for epoch in range(c.EPOCHS):
        epoch_loss, epoch_acc = train(dataloader, c.N_SAMPLES, model,
                                      loss_function, optimizer, c.DEVICE)

        losses.append(epoch_loss)
        accs.append(epoch_acc)

        if epoch % 100 == 0:
            print(f"EPOCH: {epoch + 1}", end="\t")
            print(f"Accuracy: {epoch_acc}", end="\t")
            print(f"Loss: {epoch_loss}")

    # Save Model
    torch.save(model.state_dict(), c.PATH)

    # Predict
    X_db = X_db.tensors[0].to(c.DEVICE)
    # 1번 방법(faster) torch.no_grad() autograd engine 비활성화
    with torch.no_grad():
        pred = model(X_db).to("cpu")

    # 2번 방법: model에서 바로 예측 후 detach()하여 모델의 예측값만 받기
    # pred = model(X_db).to("cpu")
    # pred = pred.to("cpu").detach()

    X_db = X_db.to("cpu").detach().numpy()
    pred = (pred > 0.5).type(torch.float).numpy()

    # Visualization
    vis_losses_accs(losses, accs)
    vis_meshgrid(X_db, pred, X, y)


def xor_with_mlp_eval(c):
    '''torch 모델 불러와서 추론만 하기'''
    # Data
    _, dataset, X_db, X, y = get_dataset(N_SAMPLES=c.N_SAMPLES, BATCH_SIZE=c.BATCH_SIZE, SEED=c.SEED)

    # Model
    model = MLP(n_features=c.n_features, n_hdn_nrns=2).to(c.DEVICE)
    model.load_state_dict(torch.load(c.PATH))
    model.eval()

    # Predict
    X_db = X_db.tensors[0].to(c.DEVICE)
    with torch.no_grad():
        pred = model(X_db).to("cpu")

    X_db = X_db.to("cpu").detach().numpy()
    pred = (pred > 0.5).type(torch.float).numpy()

    # Visualization
    vis_meshgrid(X_db, pred, X, y)



if __name__ == '__main__':
    constants = Constants(
        N_SAMPLES=300,
        BATCH_SIZE=8,
        EPOCHS=1000,
        LR=0.01,
        n_features=2,
        DEVICE=get_device(),
        PATH="model/xor_params.pt",
        SEED=8
    )
    np.random.seed(constants.SEED)
    # xor_with_mlp(constants)
    xor_with_mlp_eval(constants)





