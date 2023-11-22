from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch
from torch.optim import SGD
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from tqdm import tqdm


def show_mnist_img():
    dataset = MNIST(root='data', train=True, download=True)

    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    axes = axes.flatten()

    for i, (img, label) in enumerate(dataset):
        print(type(img))
        print(type(label))
        # img.show()
        axes[i].imshow(img, cmap="gray")
        axes[i].axis("off")
        axes[i].set_title(f"Class {label}", fontsize=15)
        img = np.array(img)
        print(img.shape, img.dtype) # uint8 unsigned integer 8 bit

        if i == 9:
            break

    fig.tight_layout()
    plt.show()


class MNIST_Classifier(nn.Module):
    def __init__(self):
        super(MNIST_Classifier, self).__init__()

        self.fc1 = nn.Linear(in_features=784, out_features=512)
        self.fc1_act = nn.ReLU()

        self.fc2 = nn.Linear(in_features=512, out_features=128)
        self.fc2_act = nn.ReLU()

        self.fc3 = nn.Linear(in_features=128, out_features=52)
        self.fc3_act = nn.ReLU()

        self.fc4 = nn.Linear(in_features=52, out_features=10)

    def forward(self, x):
        x = self.fc1_act(self.fc1(x))
        x = self.fc2_act(self.fc2(x))
        x = self.fc3_act(self.fc3(x))
        x = self.fc4(x)
        return x


def plot_metrics(losses, accs):
    fig, axes = plt.subplots(2, 1, figsize=(10, 5))

    axes[0].plot(losses)
    axes[1].plot(accs)

    axes[1].set_xlabel("Epoch", fontsize=15)
    axes[0].set_ylabel("Loss", fontsize=15)
    axes[1].set_ylabel("Accuracy", fontsize=15)
    axes[0].tick_params(labelsize=10)
    axes[1].tick_params(labelsize=10)
    fig.suptitle("MNIST 4-layer Model Metrics by Epoch", fontsize=16)
    plt.show()


def mnist_main_routine():
    BATCH_SIZE = 32
    LR = 0.003
    EPOCHS = 10

    dataset = MNIST(root='data', train=True, download=True, transform=ToTensor())
    n_samples = len(dataset)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{DEVICE=}")
    model = MNIST_Classifier().to(DEVICE)
    loss_function = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=LR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

    losses, accs = [], []
    for epoch in range(EPOCHS):
        epoch_loss, n_corrects = 0., 0

        for X_, y_ in tqdm(dataloader):
            X_, y_ = X_.to(DEVICE), y_.to(DEVICE)
            X_ = X_.reshape(BATCH_SIZE, -1)

            pred = model(X_)
            loss = loss_function(pred, y_)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(X_)
            n_corrects += (torch.max(pred, dim=1)[1] == y_).sum().item()

        epoch_loss /= n_samples
        losses.append(epoch_loss)

        epoch_acc = n_corrects / n_samples
        accs.append(epoch_acc)

        print(f"Epoch: {epoch+1}", end="\t")
        print(f"Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}")

    plot_metrics(losses, accs)


if __name__ == '__main__':
    mnist_main_routine()

