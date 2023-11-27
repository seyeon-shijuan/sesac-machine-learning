import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from tqdm import tqdm

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.cnn1_act = nn.Tanh()
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.cnn2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.cnn2_act = nn.Tanh()
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.cnn3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)
        self.cnn3_act = nn.Tanh()

        self.fc1 = nn.Linear(in_features=120, out_features=84)
        self.fc1_act = nn.Tanh()

        self.fc2 = nn.Linear(in_features=84, out_features=10)
        # self.out = nn.Softmax(dim=-1)

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


def random_test_lenet():
    model = LeNet()
    # B, C, H, W
    rand_test = torch.randn((10, 1, 28, 28))
    rst = model(rand_test)

    if rst.shape == (10, 10):
        print("test passed")


def run_lenet():
    # random test
    # random_test_lenet()

    # config
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(model)
    EPOCHS = 3
    LR = 0.001
    BATCH_SIZE = 64
    N_SAMPLES = 60000

    # MNIST config
    dataset = MNIST(root='data', train=True, download=True, transform=ToTensor())
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

    model = LeNet().to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LR)

    losses, accs = list(), list()

    for e in range(EPOCHS):
        epoch_loss, n_corrects = 0., 0

        for X_, y_ in tqdm(dataloader):
            X_, y_ = X_.to(DEVICE), y_.to(DEVICE)
            pred = model(X_)
            loss = loss_fn(pred, y_)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss
            pred_cls = torch.argmax(pred, dim=1)
            n_corrects += (pred_cls == y_).sum().item()

        epoch_loss /= N_SAMPLES
        epoch_accr = n_corrects / N_SAMPLES

        print(f"epoch {e} : loss={epoch_loss.item():.4f}, accr={epoch_accr}")

    losses.append(epoch_loss)
    accs.append(epoch_accr)
    print(losses)
    print(accs)



if __name__ == '__main__':
    run_lenet()