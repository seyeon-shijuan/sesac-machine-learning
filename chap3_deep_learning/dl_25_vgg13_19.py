import torch
from torch import nn
from collections import OrderedDict
from torchsummary import summary


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
            ('conv3-256-3', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)),
            ('conv3-256-3-act', nn.ReLU()),
            ('conv3-256-4', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)),
            ('conv3-256-4-act', nn.ReLU()),
            ('maxpool3', nn.MaxPool2d(kernel_size=2, stride=2))
        ]))

        self.conv4 = nn.Sequential(OrderedDict([
            ('conv3-512-1', nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)),
            ('conv3-512-1-act', nn.ReLU()),
            ('conv3-512-2', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)),
            ('conv3-512-2-act', nn.ReLU()),
            ('conv3-512-3', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)),
            ('conv3-512-3-act', nn.ReLU()),
            ('conv3-512-4', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)),
            ('conv3-512-4-act', nn.ReLU()),
            ('maxpool4', nn.MaxPool2d(kernel_size=2, stride=2))
        ]))

        self.conv5 = nn.Sequential(OrderedDict([
            ('conv3-512-3', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)),
            ('conv3-512-3-act', nn.ReLU()),
            ('conv3-512-4', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)),
            ('conv3-512-4-act', nn.ReLU()),
            ('conv3-512-5', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)),
            ('conv3-512-5-act', nn.ReLU()),
            ('conv3-512-6', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)),
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


if __name__ == '__main__':
    # run_vgg13()
    run_vgg19()