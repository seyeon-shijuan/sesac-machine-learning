import numpy as np
import torch
import torch.nn as nn


def conv_test():
    input_data = np.arange(6*6*3).reshape((6, 6, 3))
    # print(input_data)
    filter_data = np.arange(3*3*3).reshape((3, 3, 3))
    print(filter_data)


def run_conv1():
    H, W = 100, 150
    input_tensor = torch.randn(size=(1, H, W))

    conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3)
    output_tensor = conv(input_tensor)
    print(output_tensor.shape)


if __name__ == '__main__':
    conv_test()
    # run_conv1()
