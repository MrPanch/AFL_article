import math
import torch
from torch import nn
import torch.nn.functional as F
from models.layers import *


class Generator(nn.Module):
    def __init__(self, scale_factor, crop_size, AFL_type):
        self.scale_factor = scale_factor
        self.crop_size = crop_size
        self.AFL_type = AFL_type

        upsample_block_num = int(math.log(scale_factor, 2))

        super(Generator, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        block8 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, 3, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

        if AFL_type == 'no_AFL':
            self.fl = None
        elif AFL_type == 'G_log':
            self.fl = GeneralFourier2d((3, crop_size // scale_factor, crop_size // scale_factor), log=True)
        elif AFL_type == 'G_no_log':
            self.fl = GeneralFourier2d((3, crop_size // scale_factor, crop_size // scale_factor), log=False)
        elif AFL_type == 'L_log':
            self.fl = LinearFourier2d((3, crop_size // scale_factor, crop_size // scale_factor), log=True)
        elif AFL_type == 'L_no_log':
            self.fl = LinearFourier2d((3, crop_size // scale_factor, crop_size // scale_factor), log=False)

    def forward(self, x):
        if self.fl is not None:
            x = self.fl(x)
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        return (torch.tanh(block8) + 1) / 2


class eGenerator(nn.Module):
    def __init__(self, scale_factor, crop_size, AFL_type, num_rrdb_blocks: int = 8):
        r""" This is an esrgan model defined by the author himself.
        We use two settings for our generator – one of them contains 8 residual blocks, with a capacity similar
        to that of SRGAN and the other is a deeper model with 16/23 RRDB blocks.
        Args:
            num_rrdb_blocks (int): How many residual in residual blocks are combined. (Default: 16).
        Notes:
            Use `num_rrdb_blocks` is 16 for TITAN 2080Ti.
            Use `num_rrdb_blocks` is 23 for Tesla A100.
        """
        self.scale_factor = scale_factor
        self.crop_size = crop_size
        self.AFL_type = AFL_type
        super(eGenerator, self).__init__()

        if AFL_type == 'no_AFL':
            self.fl = None
        elif AFL_type == 'G_log':
            self.fl = GeneralFourier2d((3, crop_size // scale_factor, crop_size // scale_factor), log=True)
        elif AFL_type == 'G_no_log':
            self.fl = GeneralFourier2d((3, crop_size // scale_factor, crop_size // scale_factor), log=False)
        elif AFL_type == 'L_log':
            self.fl = LinearFourier2d((3, crop_size // scale_factor, crop_size // scale_factor), log=True)
        elif AFL_type == 'L_no_log':
            self.fl = LinearFourier2d((3, crop_size // scale_factor, crop_size // scale_factor), log=False)

        # First layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

        # 16/23 ResidualInResidualDenseBlock layer.
        trunk = []
        for _ in range(num_rrdb_blocks):
            trunk += [ResidualInResidualDenseBlock(channels=64, growth_channels=32, scale_ratio=0.2)]
        self.trunk = nn.Sequential(*trunk)

        # Second conv layer post residual blocks
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # Upsampling layers
        self.up1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.up2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # Next layer after upper sampling
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        # Final output layer
        self.conv4 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fl is not None:
            x = self.fl(x)
        out1 = self.conv1(x)
        trunk = self.trunk(out1)
        out2 = self.conv2(trunk)
        out = torch.add(out1, out2)
        out = F.leaky_relu(self.up1(F.interpolate(out, scale_factor=2, mode="nearest")), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.up2(F.interpolate(out, scale_factor=2, mode="nearest")), negative_slope=0.2, inplace=True)
        out = self.conv3(out)
        out = self.conv4(out)

        return (torch.tanh(out) + 1) / 2
