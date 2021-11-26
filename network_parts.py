import torch
import torch.nn as nn

class res_unit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.conv(self.bn(x)) + self.conv1(self.bn(x))

class res_unit_no_bn(nn.Module): # no batchnormalizing
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.conv(x) + self.conv1(x)

class down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.AvgPool2d(2, 2),
            res_unit(in_channels, out_channels)
        )
    def forward(self, x):
        return self.conv(x)

class up(nn.Module):
    def __init__(self, in_channels_1, in_channels_2, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels_1, in_channels_1 // 2, kernel_size=2, stride=2)
        self.conv = res_unit(in_channels_1 // 2 + in_channels_2, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)