import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np
from torch.utils.data import DataLoader

from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor



class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()

        self.feature = nn.Sequential(OrderedDict([
            # 1. input (224 x 224x RGB image)
            ('conv3-64', nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)),
            ('conv3-64-act', nn.ReLU()),
            ('maxpool1', nn.MaxPool2d(kernel_size=2, stride=2)),

            # 2.
            ('conv3-128', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)),
            ('conv3-128-act', nn.ReLU()),
            ('maxpool2', nn.MaxPool2d(kernel_size=2, stride=2)),

            # 3.
            ('conv3-256-1', nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)),
            ('conv3-256-1-act', nn.ReLU()),
            ('conv3-256-2', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)),
            ('conv3-256-2-act', nn.ReLU()),
            ('maxpool3', nn.MaxPool2d(kernel_size=2, stride=2)),

            # 4.
            ('conv3-512-1', nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)),
            ('conv3-512-1-act', nn.ReLU()),
            ('conv3-512-2', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)),
            ('conv3-512-2-act', nn.ReLU()),
            ('maxpool4', nn.MaxPool2d(kernel_size=2, stride=2)),

            # 5.
            ('conv3-512-3', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)),
            ('conv3-512-3-act', nn.ReLU()),
            ('conv3-512-4', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)),
            ('conv3-512-4-act', nn.ReLU()),
            ('maxpool5', nn.MaxPool2d(kernel_size=2, stride=2))
        ]))

        # (Batch ,C , h W) -> (Batch , x)
        self.classifier = nn.Sequential(OrderedDict([
            ('fc-4096-1', nn.Linear(in_features=512*7*7, out_features=4096)),
            ('fc-4096-1-act', nn.ReLU()),
            ('fc-4096-2', nn.Linear(in_features=4096, out_features=4096)),
            ('fc-4096-2-act', nn.ReLU()),
            ('fc-1000', nn.Linear(in_features=4096, out_features=1000)),
            ('fc-1000-act', nn.ReLU())
        ]))

        # 64, 512, 7, 7
        # (64, b)

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x







def run_vggnet():
    dataset = CIFAR10(root='data', train=True, transform=ToTensor(), download=True)
    dataloader = DataLoader(dataset, batch_size=16)

    test_data = torch.randn((10, 3, 224, 224))
    model = VGGNet()
    print(model)
    pred = model.forward(test_data)
    print(pred)



if __name__ == '__main__':
    run_vggnet()