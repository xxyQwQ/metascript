import torch
from torch import nn


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 4, stride: int = 2, padding: int = 1):
        super(DownsampleBlock, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, input):
        output = self.network(input)
        return output


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 4, stride: int = 2, padding: int = 1):
        super(UpsampleBlock, self).__init__()
        self.network = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, input, injection=None):
        output = self.network(input)
        if injection is not None:
            output = torch.cat((output, injection), dim=1)
        return output
