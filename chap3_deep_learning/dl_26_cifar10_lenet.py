from dataclasses import dataclass
import pickle
import csv
from torchvision.datasets import CIFAR10
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from torchvision.transforms import ToTensor
from tqdm import tqdm
import matplotlib.pyplot as plt

@dataclass
class Constants:
    N_SAMPLES: int
    BATCH_SIZE: int
    EPOCHS: int
    LR: float
    DEVICE: torch.device
    PATH: str
    METRIC_PATH: str
    SEED: int


def get_device():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"curr device = {DEVICE}")
    return DEVICE


class LeNet(nn.Module):
    def __init__(self, init_channel, out_features):
        super(LeNet, self).__init__()
        # self.cnn1 = nn.Conv2d(in_channels=init_channel, out_channels=6, kernel_size=5, padding=2)
        self.cnn1 = nn.Conv2d(in_channels=init_channel, out_channels=6, kernel_size=5, padding=0)
        self.cnn1_act = nn.Tanh()
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.cnn2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.cnn2_act = nn.Tanh()
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.cnn3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)
        self.cnn3_act = nn.Tanh()

        # self.fc1 = nn.Linear(in_features=120*2*2, out_features=84)
        self.fc1 = nn.Linear(in_features=120, out_features=84)
        self.fc1_act = nn.Tanh()

        self.fc2 = nn.Linear(in_features=84, out_features=out_features)

    def forward(self, x):
        x = self.cnn1(x)
        x = self.cnn1_act(x)
        x = self.avgpool1(x)
        x = self.cnn2(x)
        x = self.cnn2_act(x)
        x = self.avgpool2(x)
        x = self.cnn3(x)
        x = self.cnn3_act(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc1_act(x)
        x = self.fc2(x)
        return x


def train_cifar10_w_lenet(c):
    # CIFAR10 config
    dataset = CIFAR10(root='data', train=True, transform=ToTensor(), download=True)
    dataloader = DataLoader(dataset, batch_size=c.BATCH_SIZE, shuffle=True)

    model = LeNet(init_channel=3, out_features=10).to(c.DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=c.LR)

    losses, accs = list(), list()

    for e in range(c.EPOCHS):
        epoch_loss, n_corrects = 0., 0

        for X_, y_ in tqdm(dataloader):
            optimizer.zero_grad()

            X_, y_ = X_.to(c.DEVICE), y_.to(c.DEVICE)
            pred = model(X_)
            loss = loss_fn(pred, y_)

            loss.backward()
            optimizer.step()

            epoch_loss += loss
            pred_cls = torch.argmax(pred, dim=1)
            n_corrects += (pred_cls == y_).sum().item()

        epoch_loss /= len(dataloader)
        # epoch_loss /= c.N_SAMPLES
        epoch_accr = n_corrects / c.N_SAMPLES

        print(f"\n epoch {e} : loss={epoch_loss.item():.4f}, accr={epoch_accr}")

        losses.append(epoch_loss.item())
        accs.append(epoch_accr)

        if e in [199, 399, 599, 799]:
            rep = c.PATH.replace(".pt", f"_ep{e}.pt")
            torch.save(model, rep)

    # print(losses)
    # print(accs)

    # Save Model and Metrics by Epoch
    with open(c.METRIC_PATH, 'wb') as f:
        result = {
            'losses': losses,
            'accs': accs
        }
        pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)

    torch.save(model, c.PATH)

    with open("model/lenet5_metrics_2.csv", 'w') as file:
        writer = csv.writer(file)
        writer.writerow(losses)
        writer.writerow(accs)

    visualize(losses, accs)


def visualize(losses, accs):
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 5))
    axes[0].plot(losses)
    axes[1].plot(accs)

    axes[1].set_xlabel("Epoch", fontsize=15)
    axes[0].set_ylabel("Loss", fontsize=15)
    axes[1].set_ylabel("Accuracy", fontsize=15)
    axes[0].tick_params(labelsize=10)
    axes[1].tick_params(labelsize=10)
    fig.suptitle("Lenet5 Metrics by Epoch", fontsize=16)
    plt.show()


if __name__ == '__main__':
    constants = Constants(
        N_SAMPLES=50000,
        BATCH_SIZE=1024,
        EPOCHS=1000,
        LR=0.0001,
        DEVICE=get_device(),
        PATH="model/lenet5_cifar10_2.pt",
        METRIC_PATH="model/lenet5_metrics_2.pkl",
        SEED=80
    )
    train_cifar10_w_lenet(constants)

