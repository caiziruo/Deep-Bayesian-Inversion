import torch.nn as nn
from network_parts import *


class Generator(nn.Module):
    def __init__(self, in_channels, out_channels = 1):
        super(Generator, self).__init__()

        self.inc = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            res_unit_no_bn(32, 32)
        )
        self.down1 = down(32, 64)
        self.down2 = down(64, 128)
        self.down3 = down(128, 256)
        self.down4 = down(256, 512)
        self.down5 = down(512, 512)
        self.down6 = down(512, 512)
        self.res = res_unit(512, 512)
        self.up1 = up(512, 512, 512)
        self.up2 = up(512, 512, 512)
        self.up3 = up(512, 256, 256)
        self.up4 = up(256, 128, 128)
        self.up5 = up(128, 64, 64)
        self.up6 = up(64, 32, 32)
        self.outc = nn.Sequential(
            res_unit_no_bn(32, 32),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x = self.down6(x6)
        x = self.res(x)
        x = self.up1(x, x6)
        x = self.up2(x, x5)
        x = self.up3(x, x4)
        x = self.up4(x, x3)
        x = self.up5(x, x2)
        x = self.up6(x, x1)
        output = self.outc(x)
        return output


