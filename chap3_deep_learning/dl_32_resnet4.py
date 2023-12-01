import torch.nn as nn
import torch


class ResNet(nn.Module):
    def __init__(self, in_channels, n_blocks_list):
        super(ResNet, self).__init__()
        # 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64,
                      kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.conv2 = self._make_layers(in_channels=64, out_channels=64, n_blocks=n_blocks_list[0])
        self.conv3 = self._make_layers(in_channels=64, out_channels=128, n_blocks=n_blocks_list[1], downsample=True)
        self.conv4 = self._make_layers(in_channels=128, out_channels=256, n_blocks=n_blocks_list[2], downsample=True)
        self.conv5 = self._make_layers(in_channels=256, out_channels=512, n_blocks=n_blocks_list[3], downsample=True)

        # 6
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=7)
        self.fc = nn.Linear(in_features=512, out_features=1000)

    @staticmethod
    def _make_layers(in_channels, out_channels, n_blocks, downsample=False):
        layers = list()

        if downsample:
            layers.append(ResidualBlock(in_channels=in_channels, out_channels=out_channels, stride=2))
            in_channels = out_channels
            n_blocks -= 1

        for _ in range(n_blocks):
            layers.append(ResidualBlock(in_channels=in_channels, out_channels=out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
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


def ResNet18(): return ResNet(in_channels=3, n_blocks_list=[2, 2, 2, 2])


def ResNet32(): return ResNet(in_channels=3, n_blocks_list=[3, 4, 6, 3])


def run_resnet():
    input_tensor = torch.randn(size=(32, 3, 224, 224))
    model = ResNet18()
    pred = model(input_tensor)
    print(pred.shape)

    model2 = ResNet32()
    pred2 = model2(input_tensor)
    print(pred2.shape)


if __name__ == '__main__':
    run_resnet()

