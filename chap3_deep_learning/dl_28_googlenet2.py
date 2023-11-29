from dataclasses import dataclass
import torch.nn as nn
import torch
from torch.optim import Adam
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from chap3_deep_learning.dl_27_googlenet1 import Inception
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv

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


def save_metrics(filepath, n_epochs, loss, accuracy):
    epochs = [e for e in range(n_epochs)]
    with open(filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Accuracy', 'Loss'])
        for e, acc, l in zip(epochs, accuracy, loss):
            writer.writerow([e, acc, l])

def visualize(losses, accrs):
    fig, axes = plt.subplots(2, 1, figsize=(15, 7))
    axes[0].plot(losses)
    axes[1].plot(accrs)
    axes[1].set_xlabel('EPOCHS', fontsize=15)
    axes[0].set_ylabel('Loss', fontsize=15)
    axes[1].set_ylabel('Accuracy', fontsize=15)
    axes[0].tick_params(labelsize=15)
    axes[1].tick_params(labelsize=15)
    fig.suptitle("GoogLeNet Metrics by Epoch", fontsize=16)
    plt.savefig('model/googlenet_vis.png')
    plt.show()


class GoogLeNet(nn.Module):
    def __init__(self, in_channels):
        super(GoogLeNet, self).__init__()
        # 1
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, padding=3, stride=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 2
        self.conv2_red = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0, stride=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1, stride=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 3
        self.icp_3a = Inception(in_channels=192, ch1x1=64, ch3x3red=96, ch3x3=128,
                                ch5x5red=16, ch5x5=32, pool_proj=32)
        self.icp_3b = Inception(in_channels=256, ch1x1=128, ch3x3red=128, ch3x3=192,
                                ch5x5red=32, ch5x5=96, pool_proj=64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 4
        self.icp_4a = Inception(in_channels=480, ch1x1=192, ch3x3red=96, ch3x3=208,
                                ch5x5red=16, ch5x5=48, pool_proj=64)
        self.icp_4b = Inception(in_channels=512, ch1x1=160, ch3x3red=112, ch3x3=224,
                                ch5x5red=24, ch5x5=64, pool_proj=64)
        self.icp_4c = Inception(in_channels=512, ch1x1=128, ch3x3red=128, ch3x3=256,
                                ch5x5red=24, ch5x5=64, pool_proj=64)
        self.icp_4d = Inception(in_channels=512, ch1x1=112, ch3x3red=144, ch3x3=288,
                                ch5x5red=32, ch5x5=64, pool_proj=64)
        self.icp_4e = Inception(in_channels=528, ch1x1=256, ch3x3red=160, ch3x3=320,
                                ch5x5red=32, ch5x5=128, pool_proj=128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 5
        # self.icp_5a = Inception(in_channels=832, ch1x1=256, ch3x3red=160, ch3x3=320,
        #                         ch5x5red=32, ch5x5=128, pool_proj=128)
        # self.icp_5b = Inception(in_channels=832, ch1x1=384, ch3x3red=192, ch3x3=384,
        #                         ch5x5red=48, ch5x5=128, pool_proj=128)
        # self.avgpool5 = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

        # 6
        self.fc6 = nn.Linear(in_features=832, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)

        x = self.conv2_red(x)
        x = self.conv2(x)
        x = self.maxpool2(x)

        x = self.icp_3a(x)
        x = self.icp_3b(x)
        x = self.maxpool3(x)

        x = self.icp_4a(x)
        x = self.icp_4b(x)
        x = self.icp_4c(x)
        x = self.icp_4d(x)
        x = self.icp_4e(x)
        x = self.maxpool4(x)

        # x = self.icp_5a(x)
        # x = self.icp_5b(x)
        # x = self.avgpool5(x)

        x = x.view(x.size(0), -1)
        x = self.fc6(x)
        return x


def train_cifar10(c):
    data = CIFAR10(root='data', download=True, train=True, transform=ToTensor())
    dataloader = DataLoader(data, batch_size=c.BATCH_SIZE, shuffle=True)
    model = GoogLeNet(in_channels=3)
    model = model.to(c.DEVICE)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=c.LR)

    losses, accs = list(), list()
    for e in range(c.EPOCHS):
        epoch_loss, n_corrects = 0., 0
        for X_, y_ in tqdm(dataloader):
            X_, y_ = X_.to(c.DEVICE), y_.to(c.DEVICE)
            pred = model(X_)

            optimizer.zero_grad()
            loss = loss_fn(pred, y_)
            loss.backward()
            optimizer.step()

            pred_cls = torch.argmax(pred, dim=1)
            n_corrects += (pred_cls == y_).sum().item()
            epoch_loss += loss

        epoch_loss /= len(dataloader)
        epoch_accr = n_corrects / c.N_SAMPLES
        losses.append(epoch_loss.item())
        accs.append(epoch_accr)

        print(f"epoch {e}- loss: {epoch_loss.item():.4f}, accuracy: {epoch_accr:.4f}")

        if e in [299, 499, 699, 899]:
            torch.save(model, c.PATH.replace('.pt', f"_epoch_{e}.pt"))

    save_metrics(filepath=c.METRIC_PATH, n_epochs=c.EPOCHS, loss=losses, accuracy=accs)
    torch.save(model, c.PATH)
    visualize(losses=losses, accrs=accs)


if __name__ == '__main__':
    constants = Constants(
        N_SAMPLES=50000,
        BATCH_SIZE=2048,
        EPOCHS=1000,
        LR=0.0001,
        DEVICE=get_device(),
        PATH='model/googlenet_cifar10.pt',
        METRIC_PATH='model/googlenet_cifar10_metric.csv',
        SEED=80
    )

    train_cifar10(constants)