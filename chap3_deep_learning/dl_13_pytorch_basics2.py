from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import SGD


class MLP(nn.Module):
    def __init__(self, n_features, n_hdn_nrns):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features=n_features, out_features=n_hdn_nrns)
        self.fc1_act = nn.Sigmoid()
        self.fc2 = nn.Linear(in_features=n_hdn_nrns, out_features=1)
        self.fc2_act = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc1_act(x)
        x = self.fc2(x)
        x = self.fc2_act(x)
        x = x.view(-1)
        return x


def get_dataset(N_SAMPLES=300, BATCH_SIZE=8):
    X, y = make_moons(n_samples=N_SAMPLES, noise=0.2)
    dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
    return dataloader


def get_device():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"curr device = {DEVICE}")
    return DEVICE


def train(dataloader, N_SAMPLES, model, loss_function, optimizer, DEVICE):
    epoch_loss, n_corrects = 0., 0
    for X, y in dataloader:
        X, y = X.to(DEVICE), y.to(DEVICE)

        pred = model(X)
        loss = loss_function(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * len(X)
        pred = (pred > 0.5).type(torch.float)
        n_corrects += (pred == y).sum().item()

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
    fig.suptitle("2-layer Model Eval Metrics by Epoch", fontsize=16)
    # fig.tight_layout()
    # plt.show()


if __name__ == '__main__':

    N_SAMPLES = 300
    BATCH_SIZE = 8
    EPOCHS = 100
    LR = 0.01
    n_features = 2
    DEVICE = get_device()

    dataloader = get_dataset(N_SAMPLES, BATCH_SIZE)

    model = MLP(n_features=n_features, n_hdn_nrns=3).to(DEVICE)
    loss_function = nn.BCELoss()
    optimizer = SGD(model.parameters(), lr=LR)

    losses, accs = list(), list()

    for epoch in range(EPOCHS):
        epoch_loss, epoch_acc = train(dataloader, N_SAMPLES, model,
                                      loss_function, optimizer, DEVICE)

        losses.append(epoch_loss)
        accs.append(epoch_acc)
        print(f"EPOCH: {epoch + 1}", end="\t")
        print(f"Accuracy: {epoch_acc}", end="\t")
        print(f"Loss: {epoch_loss}")

    vis_losses_accs(losses, accs)

