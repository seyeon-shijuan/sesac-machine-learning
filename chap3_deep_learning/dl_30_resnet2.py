import torch.nn as nn
import torch


class ResNet34(nn.Module):
    def __init__(self, in_channels):
        super(ResNet34, self).__init__()
        # 1
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64,
                               kernel_size=7, stride=2, padding=3)
        self.conv1_act = nn.ReLU()

        # 2
        self.conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ResidualBlock(in_channels=64, out_channels=64),
            ResidualBlock(in_channels=64, out_channels=64),
            ResidualBlock(in_channels=64, out_channels=64))

        # 3
        self.conv3 = nn.Sequential(
            ResidualBlock(in_channels=64, out_channels=128, stride=2),
            ResidualBlock(in_channels=128, out_channels=128),
            ResidualBlock(in_channels=128, out_channels=128),
            ResidualBlock(in_channels=128, out_channels=128))

        # 4
        self.conv4 = nn.Sequential(
            ResidualBlock(in_channels=128, out_channels=256, stride=2),
            ResidualBlock(in_channels=256, out_channels=256),
            ResidualBlock(in_channels=256, out_channels=256),
            ResidualBlock(in_channels=256, out_channels=256),
            ResidualBlock(in_channels=256, out_channels=256),
            ResidualBlock(in_channels=256, out_channels=256))

        # 5
        self.conv5 = nn.Sequential(
            ResidualBlock(in_channels=256, out_channels=512, stride=2),
            ResidualBlock(in_channels=512, out_channels=512),
            ResidualBlock(in_channels=512, out_channels=512))

        # 6
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=7)
        # self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(in_features=512, out_features=1000)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1_act(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv_path = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=3, padding=1, stride=stride),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=3, padding=1, stride=1))

        if in_channels != out_channels:
            self.skip_path = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                       kernel_size=1, padding=0, stride=2)
        else:
            self.skip_path = nn.Identity()

        self.out_act = nn.ReLU()

    def forward(self, x):
        out = self.conv_path(x)
        out += self.skip_path(x)
        out = self.out_act(out)
        return out


def run_resnet():
    input_tensor = torch.randn(size=(32, 3, 224, 224))
    model = ResNet34(in_channels=3)
    pred = model(input_tensor)
    print(pred.shape)


if __name__ == '__main__':
    run_resnet()