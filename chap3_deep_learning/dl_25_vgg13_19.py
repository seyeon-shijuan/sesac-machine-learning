import pickle

import torch
from torch import nn
from collections import OrderedDict
from dataclasses import dataclass
import torch.optim as optim

from torch.utils.data import DataLoader
from torchsummary import summary
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from tqdm import tqdm


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

class VGG13(nn.Module):
    def __init__(self):
        super(VGG13, self).__init__()
        self.conv1 = nn.Sequential(OrderedDict([
            ('conv3-64-1', nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)),
            ('conv3-64-1-act', nn.ReLU()),
            ('conv3-64-2', nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)),
            ('conv3-64-2-act', nn.ReLU()),
            ('maxpool1', nn.MaxPool2d(kernel_size=2, stride=2))
        ]))

        self.conv2 = nn.Sequential(OrderedDict([
            ('conv3-128-1', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)),
            ('conv3-128-1-act', nn.ReLU()),
            ('conv3-128-2', nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)),
            ('conv3-128-2-act', nn.ReLU()),
            ('maxpool2', nn.MaxPool2d(kernel_size=2, stride=2))
        ]))

        self.conv3 = nn.Sequential(OrderedDict([
            ('conv3-256-1', nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)),
            ('conv3-256-1-act', nn.ReLU()),
            ('conv3-256-2', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)),
            ('conv3-256-2-act', nn.ReLU()),
            ('maxpool3', nn.MaxPool2d(kernel_size=2, stride=2))
        ]))

        self.conv4 = nn.Sequential(OrderedDict([
            ('conv3-512-1', nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)),
            ('conv3-512-1-act', nn.ReLU()),
            ('conv3-512-2', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)),
            ('conv3-512-2-act', nn.ReLU()),
            ('maxpool4', nn.MaxPool2d(kernel_size=2, stride=2))
        ]))

        self.conv5 = nn.Sequential(OrderedDict([
            ('conv3-512-3', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)),
            ('conv3-512-3-act', nn.ReLU()),
            ('conv3-512-4', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)),
            ('conv3-512-4-act', nn.ReLU()),
            ('maxpool5', nn.MaxPool2d(kernel_size=2, stride=2))
        ]))

        self.classifier = nn.Sequential(OrderedDict([
            ('fc-4096-1', nn.Linear(in_features=512*7*7, out_features=4096)),
            ('fc-4096-1-act', nn.ReLU()),
            ('fc-4096-2', nn.Linear(in_features=4096, out_features=4096)),
            ('fc-4096-2-act', nn.ReLU()),
            ('fc-1000', nn.Linear(in_features=4096, out_features=1000))
        ]))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        self.conv1 = nn.Sequential(OrderedDict([
            ('conv3-64-1', nn.Conv2d(in_channels=3, out_channels=64,
                                     kernel_size=3, padding=1)),
            ('conv3-64-1-act', nn.ReLU()),
            ('conv3-64-2', nn.Conv2d(in_channels=64, out_channels=64,
                                     kernel_size=3, padding=1)),
            ('conv3-64-2-act', nn.ReLU()),
            ('maxpool1', nn.MaxPool2d(kernel_size=2, stride=2))
        ]))

        self.conv2 = nn.Sequential(OrderedDict([
            ('conv3-128-1', nn.Conv2d(in_channels=64, out_channels=128,
                                      kernel_size=3, padding=1)),
            ('conv3-128-1-act', nn.ReLU()),
            ('conv3-128-2', nn.Conv2d(in_channels=128, out_channels=128,
                                      kernel_size=3, padding=1)),
            ('conv3-128-2-act', nn.ReLU()),
            ('maxpool2', nn.MaxPool2d(kernel_size=2, stride=2))
        ]))

        self.conv3 = nn.Sequential(OrderedDict([
            ('conv3-256-1', nn.Conv2d(in_channels=128, out_channels=256,
                                      kernel_size=3, padding=1)),
            ('conv3-256-1-act', nn.ReLU()),
            ('conv3-256-2', nn.Conv2d(in_channels=256, out_channels=256,
                                      kernel_size=3, padding=1)),
            ('conv3-256-2-act', nn.ReLU()),
            ('conv3-256-3', nn.Conv2d(in_channels=256, out_channels=256,
                                      kernel_size=3, padding=1)),
            ('conv3-256-3-act', nn.ReLU()),
            ('conv3-256-4', nn.Conv2d(in_channels=256, out_channels=256,
                                      kernel_size=3, padding=1)),
            ('conv3-256-4-act', nn.ReLU()),
            ('maxpool3', nn.MaxPool2d(kernel_size=2, stride=2))
        ]))

        self.conv4 = nn.Sequential(OrderedDict([
            ('conv3-512-1', nn.Conv2d(in_channels=256, out_channels=512,
                                      kernel_size=3, padding=1)),
            ('conv3-512-1-act', nn.ReLU()),
            ('conv3-512-2', nn.Conv2d(in_channels=512, out_channels=512,
                                      kernel_size=3, padding=1)),
            ('conv3-512-2-act', nn.ReLU()),
            ('conv3-512-3', nn.Conv2d(in_channels=512, out_channels=512,
                                      kernel_size=3, padding=1)),
            ('conv3-512-3-act', nn.ReLU()),
            ('conv3-512-4', nn.Conv2d(in_channels=512, out_channels=512,
                                      kernel_size=3, padding=1)),
            ('conv3-512-4-act', nn.ReLU()),
            ('maxpool4', nn.MaxPool2d(kernel_size=2, stride=2))
        ]))

        self.conv5 = nn.Sequential(OrderedDict([
            ('conv3-512-3', nn.Conv2d(in_channels=512, out_channels=512,
                                      kernel_size=3, padding=1)),
            ('conv3-512-3-act', nn.ReLU()),
            ('conv3-512-4', nn.Conv2d(in_channels=512, out_channels=512,
                                      kernel_size=3, padding=1)),
            ('conv3-512-4-act', nn.ReLU()),
            ('conv3-512-5', nn.Conv2d(in_channels=512, out_channels=512,
                                      kernel_size=3, padding=1)),
            ('conv3-512-5-act', nn.ReLU()),
            ('conv3-512-6', nn.Conv2d(in_channels=512, out_channels=512,
                                      kernel_size=3, padding=1)),
            ('conv3-512-6-act', nn.ReLU()),
            ('maxpool5', nn.MaxPool2d(kernel_size=2, stride=2))
        ]))

        self.classifier = nn.Sequential(OrderedDict([
            ('fc-4096-1', nn.Linear(in_features=512*7*7, out_features=4096)),
            ('fc-4096-1-act', nn.ReLU()),
            ('fc-4096-2', nn.Linear(in_features=4096, out_features=4096)),
            ('fc-4096-2-act', nn.ReLU()),
            ('fc-1000', nn.Linear(in_features=4096, out_features=1000))
        ]))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ConvBlockBase(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers):
        super(ConvBlockBase, self).__init__()

        self.layers = [
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=3, padding=1),
            nn.ReLU()
        ]

        for _ in range(n_layers -1):
            self.layers.append(nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                                         kernel_size=3, padding=1))
            self.layers.append(nn.ReLU())

        # 마지막에 max pooling 추가
        self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # list에 들어있는 layer을 풀어 nn.Sequential에 입력
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.layers(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers):
        super(ConvBlock, self).__init__()

        self.layers = list()

        for i in range(n_layers):
            self.layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=3, padding=1))
            self.layers.append(nn.ReLU())
            in_channels = out_channels

        # 마지막에 max pooling 추가
        self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # list에 들어있는 layer을 풀어 nn.Sequential에 입력
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.layers(x)
        return x


class VGG11Block(nn.Module):
    def __init__(self):
        super(VGG11Block, self).__init__()
        self.conv1 = ConvBlock(in_channels=3, out_channels=64,
                               n_layers=1)
        self.conv2 = ConvBlock(in_channels=64, out_channels=128,
                               n_layers=1)
        self.conv3 = ConvBlock(in_channels=128, out_channels=256,
                               n_layers=2)
        self.conv4 = ConvBlock(in_channels=256, out_channels=512,
                               n_layers=2)
        self.conv5 = ConvBlock(in_channels=512, out_channels=512,
                               n_layers=2)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=512 * 7 * 7, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=1000)
        )

    def forward(self, x):
        x = self.conv1.forward(x)
        x = self.conv2.forward(x)
        x = self.conv3.forward(x)
        x = self.conv4.forward(x)
        x = self.conv5.forward(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class VGG13Block(nn.Module):
    def __init__(self):
        super(VGG13Block, self).__init__()

        self.conv1 = ConvBlock(in_channels=3, out_channels=64,
                               n_layers=2)
        self.conv2 = ConvBlock(in_channels=64, out_channels=128,
                               n_layers=2)
        self.conv3 = ConvBlock(in_channels=128, out_channels=256,
                               n_layers=2)
        self.conv4 = ConvBlock(in_channels=256, out_channels=512,
                               n_layers=2)
        self.conv5 = ConvBlock(in_channels=512, out_channels=512,
                               n_layers=2)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=512*7*7, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=1000)
        )

    def forward(self, x):
        x = self.conv1.forward(x)
        x = self.conv2.forward(x)
        x = self.conv3.forward(x)
        x = self.conv4.forward(x)
        x = self.conv5.forward(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class VGG19Block(nn.Module):
    def __init__(self):
        super(VGG19Block, self).__init__()

        self.conv1 = ConvBlock(in_channels=3, out_channels=64,
                               n_layers=2)
        self.conv2 = ConvBlock(in_channels=64, out_channels=128,
                               n_layers=2)
        self.conv3 = ConvBlock(in_channels=128, out_channels=256,
                               n_layers=4)
        self.conv4 = ConvBlock(in_channels=256, out_channels=512,
                               n_layers=4)
        self.conv5 = ConvBlock(in_channels=512, out_channels=512,
                               n_layers=4)

        # original 512*7*7

        self.classifier = nn.Sequential(
            nn.Linear(in_features=512*1*1, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=1000)
        )

    def forward(self, x):
        x = self.conv1.forward(x)
        x = self.conv2.forward(x)
        x = self.conv3.forward(x)
        x = self.conv4.forward(x)
        x = self.conv5.forward(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def run_vgg13():
    test_data = torch.randn((8, 3, 224, 224))
    model = VGG13()
    summary(model, input_size=(3, 224, 224), batch_size=16, device='cpu')
    pred = model.forward(test_data)
    print(pred.shape)


def run_vgg19():
    test_data = torch.randn((8, 3, 224, 224))
    model = VGG19()
    summary(model, input_size=(3, 224, 224), batch_size=16, device='cpu')
    pred = model.forward(test_data)
    print(pred.shape)


def run_conv_block():
    test_data = torch.randn((8, 3, 224, 224))
    conv1 = ConvBlock(in_channels=3, out_channels=64, n_layers=2)
    x = conv1.forward(test_data)
    print(x.shape)


def run_vgg11_block():
    test_data = torch.randn((8, 3, 224, 224))
    model = VGG11Block()
    summary(model, input_size=(3, 224, 224), batch_size=16, device='cpu')
    pred = model.forward(test_data)
    print(pred.shape)


def run_vgg13_block():
    test_data = torch.randn((8, 3, 224, 224))
    model = VGG13Block()
    summary(model, input_size=(3, 224, 224), batch_size=16, device='cpu')
    pred = model.forward(test_data)
    print(pred.shape)


def run_vgg19_block():
    test_data = torch.randn((8, 3, 224, 224))
    model = VGG19Block()
    summary(model, input_size=(3, 224, 224), batch_size=16, device='cpu')
    pred = model.forward(test_data)
    print(pred.shape)



def classify_cifar10(c):
    # (50000, 32, 32, 3)
    dataset = CIFAR10(root='data', train=True, transform=ToTensor(), download=True)
    dataloader = DataLoader(dataset, batch_size=c.BATCH_SIZE)

    model = VGG19Block()

    model = model.to(c.DEVICE)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=c.LR)

    losses, accs = list(), list()

    for e in range(c.EPOCHS):
        epoch_loss, n_corrects = 0., 0
        for X_, y_ in tqdm(dataloader):
            X_, y_ = X_.to(c.DEVICE), y_.to(c.DEVICE)

            pred = model.forward(X_)
            loss = loss_fn(pred, y_)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss
            pred_cls = torch.argmax(pred, dim=1)
            n_corrects += (pred_cls == y_).sum().item()

        epoch_loss /= c.N_SAMPLES
        epoch_accr = n_corrects / c.N_SAMPLES

        print(f"epoch {e} : loss={epoch_loss.item():.4f}, accr={epoch_accr}")
        losses.append(epoch_loss)
        accs.append(epoch_accr)

    print("==============")
    print(f"{losses=}, {accs=}")

    # Save Model and Metrics by Epoch
    with open(c.METRIC_PATH, 'wb') as f:
        result = {
            'losses': losses,
            'accs': accs
        }
        pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)

    torch.save(model, c.PATH)




if __name__ == '__main__':
    # run_vgg13()
    # run_vgg19()
    # run_conv_block()
    # run_vgg11_block()
    # run_vgg13_block()
    # run_vgg19_block()
    constants = Constants(
        N_SAMPLES=50000,
        BATCH_SIZE=32,
        EPOCHS=3,
        LR=0.01,
        DEVICE=get_device(),
        PATH="model/vgg19_cifar10.pt",
        METRIC_PATH="model/vgg_cifar10_metrics.pkl",
        SEED=80
    )
    classify_cifar10(constants)
