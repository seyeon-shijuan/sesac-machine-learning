import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=kernel_size, padding=padding, stride=stride)
        self.conv1_act = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1_act(x)
        return x


class InceptionNaive(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3, ch5x5):
        # input channel 수와 1x1, 3x3, 5x5 branch의 output channel 수를 입력받음
        super(InceptionNaive, self).__init__()
        self.branch1 = ConvBlock(in_channels=in_channels, out_channels=ch1x1,
                                 kernel_size=1, padding=0)
        self.branch2 = ConvBlock(in_channels=in_channels, out_channels=ch3x3,
                                 kernel_size=3, padding=1)
        self.branch3 = ConvBlock(in_channels=in_channels, out_channels=ch5x5,
                                 kernel_size=5, padding=2)
        self.branch4 = nn.MaxPool2d(kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        concat_axis = 1 if x.dim() == 4 else 0
        out_branch1 = self.branch1.forward(x)
        out_branch2 = self.branch2.forward(x)
        out_branch3 = self.branch3.forward(x)
        out_branch4 = self.branch4.forward(x)
        to_concat = (out_branch1, out_branch2, out_branch3, out_branch4)
        x = torch.cat(tensors=to_concat, dim=concat_axis)
        return x


class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()
        self.branch1 = ConvBlock(in_channels=in_channels, out_channels=ch1x1,
                                 kernel_size=1, padding=0)
        self.branch2 = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=ch3x3red,
                      kernel_size=1, padding=0),
            ConvBlock(in_channels=ch3x3red, out_channels=ch3x3,
                      kernel_size=3, padding=1))

        self.branch3 = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=ch5x5red,
                      kernel_size=1, padding=0),
            ConvBlock(in_channels=ch5x5red, out_channels=ch5x5,
                      kernel_size=5, padding=2))

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, padding=1, stride=1),
            ConvBlock(in_channels=in_channels, out_channels=pool_proj,
                      kernel_size=1, padding=0)
        )

    def forward(self, x):
        concat_axis = 1 if x.dim() == 4 else 0
        out_branch1 = self.branch1.forward(x)
        out_branch2 = self.branch2.forward(x)
        out_branch3 = self.branch3.forward(x)
        out_branch4 = self.branch4.forward(x)
        to_concat = (out_branch1, out_branch2, out_branch3, out_branch4)
        x = torch.cat(tensors=to_concat, dim=concat_axis)
        return x


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
        self.icp_5a = Inception(in_channels=832, ch1x1=256, ch3x3red=160, ch3x3=320,
                                ch5x5red=32, ch5x5=128, pool_proj=128)
        self.icp_5b = Inception(in_channels=832, ch1x1=384, ch3x3red=192, ch3x3=384,
                                ch5x5red=48, ch5x5=128, pool_proj=128)
        self.avgpool5 = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

        # 6
        self.fc6 = nn.Linear(in_features=1024, out_features=1000)

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

        x = self.icp_5a(x)
        x = self.icp_5b(x)
        x = self.avgpool5(x)

        x = x.view(x.size(0), -1)
        x = self.fc6(x)
        return x


def run_inception_naive():
    BATCH_SIZE = 32
    H, W = 100, 100
    channels = 192
    input_tensor = torch.randn(size=(BATCH_SIZE, channels, H, W))
    inception = InceptionNaive(in_channels=192, ch1x1=64, ch3x3=128, ch5x5=32)
    output_tensor = inception(input_tensor)
    print(f"{output_tensor.shape=}")


def run_inception_dim_reduction():
    BATCH_SIZE = 32
    H, W = 100, 100
    channels = 192
    input_tensor = torch.randn(size=(BATCH_SIZE, channels, H, W))
    # in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj
    inception = Inception(in_channels=192, ch1x1=64,
                          ch3x3red=96,ch3x3=128,
                          ch5x5red=16, ch5x5=32,
                          pool_proj=32)
    output_tensor = inception(input_tensor)
    print(f"{output_tensor.shape=}")


def run_googlenet():
    BATCH_SIZE = 32
    H, W = 224, 224
    channels = 3
    input_tensor = torch.randn(size=(BATCH_SIZE, channels, H, W))
    model = GoogLeNet(in_channels=channels)
    pred = model(input_tensor)
    print(f"{pred.shape=}")



if __name__ == '__main__':
    # run_inception_naive()
    # run_inception_dim_reduction()
    run_googlenet()



