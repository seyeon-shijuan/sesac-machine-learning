import pickle
from dataclasses import dataclass
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.optim import Adam

import numpy as np
import matplotlib.pyplot as plt
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
    METRIC_PATH: str
    SEED: int


def get_device():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"curr device = {DEVICE}")
    return DEVICE


class MnistClassifier(nn.Module):
    def __init__(self):
        super(MnistClassifier, self).__init__()
        self.fc1 = nn.Linear(in_features=784, out_features=256)
        self.fc1_act = nn.ReLU()
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.fc2_act = nn.ReLU()
        self.fc3 = nn.Linear(in_features=128, out_features=64)
        self.fc3_act = nn.ReLU()
        self.fc4 = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc1_act(x)
        x = self.fc2(x)
        x = self.fc2_act(x)
        x = self.fc3(x)
        x = self.fc3_act(x)
        x = self.fc4(x)
        return x


def train(dataloader, N_SAMPLES, model, loss_function, optimizer, DEVICE):
    epoch_loss, n_corrects = 0., 0

    for X, y in tqdm(dataloader):
        X = X.view(-1, (X.shape[-1] * X.shape[-2]))

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
    fig.suptitle("MNIST 4-layer Model Metrics by Epoch", fontsize=16)
    plt.show()


def run_mnist_classifier(c):
    # Data
    dataset = MNIST(root='data', train=True, download=True, transform=ToTensor())
    dataloader = DataLoader(dataset, batch_size=c.BATCH_SIZE)

    # Model
    model = MnistClassifier().to(c.DEVICE)
    loss_function = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=c.LR)

    # Train
    losses, accs = list(), list()

    for epoch in range(c.EPOCHS):
        epoch_loss, epoch_acc = train(dataloader, len(dataset), model, loss_function,
                                      optimizer, c.DEVICE)

        losses.append(epoch_loss)
        accs.append(epoch_acc)

        print(f"\n EPOCH: {epoch}", end="\t")
        print(f"Accuracy: {epoch_acc}", end="\t")
        print(f"Loss: {epoch_loss}")

    # Save Model and Metrics by Epoch
    with open(c.METRIC_PATH, 'wb') as f:
        result = {
            'losses': losses,
            'accs': accs
        }
        pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)

    torch.save(model, c.PATH)


def eval_and_visualize(c):
    with open(c.METRIC_PATH, 'rb') as f:
        metric_dict = pickle.load(f)

    # vis_losses_accs(metric_dict['losses'], metric_dict['accs'])

    dataset = MNIST(root='data', train=False, download=True, transform=ToTensor())
    dataloader = DataLoader(dataset)

    for label_to_extract in range(10):

        selected_image, selected_label = next(
            (img, label) for img, label in dataloader if label == label_to_extract
        )
        print('here')

    model = torch.load(c.PATH)

    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(14, 14))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        # ax.imshow(im[i].reshape(28,28), cmap="gray")
        pass




if __name__ == '__main__':
    constants = Constants(
        N_SAMPLES=60000,
        BATCH_SIZE=32,
        EPOCHS=20,
        LR=0.01,
        n_features=784,
        DEVICE=get_device(),
        PATH="model/mnist.pt",
        METRIC_PATH="model/mnist_metrics.pkl",
        SEED=80
    )
    # run_mnist_classifier(constants)
    eval_and_visualize(constants)
