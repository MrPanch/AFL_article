import math
import torch
from torch import nn




class Generator(nn.Module):
    def __init__(self, scale_factor, crop_size, AFL_type):
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
    def __init__(self, num_rrdb_blocks: int = 8):
        r""" This is an esrgan model defined by the author himself.
        We use two settings for our generator – one of them contains 8 residual blocks, with a capacity similar
        to that of SRGAN and the other is a deeper model with 16/23 RRDB blocks.
        Args:
            num_rrdb_blocks (int): How many residual in residual blocks are combined. (Default: 16).
        Notes:
            Use `num_rrdb_blocks` is 16 for TITAN 2080Ti.
            Use `num_rrdb_blocks` is 23 for Tesla A100.
        """
        super(Generator, self).__init__()

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
        out1 = self.conv1(x)
        trunk = self.trunk(out1)
        out2 = self.conv2(trunk)
        out = torch.add(out1, out2)
        out = F.leaky_relu(self.up1(F.interpolate(out, scale_factor=2, mode="nearest")), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.up2(F.interpolate(out, scale_factor=2, mode="nearest")), negative_slope=0.2, inplace=True)
        out = self.conv3(out)
        out = self.conv4(out)

        return out

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),


            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)

        )

    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size))


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x


class ResidualDenseBlock(nn.Module):
    r"""The residual block structure of traditional SRGAN and Dense model is defined"""

    def __init__(self, channels: int = 64, growth_channels: int = 32, scale_ratio: float = 0.2):
        r"""
        Args:
            channels (int): Number of channels in the input image. (Default: 64)
            growth_channels (int): how many filters to add each layer (`k` in paper). (Default: 32)
            scale_ratio (float): Residual channel scaling column. (Default: 0.2)
        """
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels + 0 * growth_channels, growth_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(channels + 1 * growth_channels, growth_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(channels + 2 * growth_channels, growth_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(channels + 3 * growth_channels, growth_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.conv5 = nn.Conv2d(channels + 4 * growth_channels, channels, kernel_size=3, stride=1, padding=1)

        self.scale_ratio = scale_ratio

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv1 = self.conv1(x)
        conv2 = self.conv2(torch.cat((x, conv1), dim=1))
        conv3 = self.conv3(torch.cat((x, conv1, conv2), dim=1))
        conv4 = self.conv4(torch.cat((x, conv1, conv2, conv3), dim=1))
        conv5 = self.conv5(torch.cat((x, conv1, conv2, conv3, conv4), dim=1))

        return conv5 * self.scale_ratio + x


class ResidualInResidualDenseBlock(nn.Module):
    r"""The residual block structure of traditional ESRGAN and Dense model is defined"""

    def __init__(self, channels: int = 64, growth_channels: int = 32, scale_ratio: float = 0.2):
        r"""
        Args:
            channels (int): Number of channels in the input image. (Default: 64)
            growth_channels (int): how many filters to add each layer (`k` in paper). (Default: 32)
            scale_ratio (float): Residual channel scaling column. (Default: 0.2)
        """
        super(ResidualInResidualDenseBlock, self).__init__()
        self.RDB1 = ResidualDenseBlock(channels, growth_channels, scale_ratio)
        self.RDB2 = ResidualDenseBlock(channels, growth_channels, scale_ratio)
        self.RDB3 = ResidualDenseBlock(channels, growth_channels, scale_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)

        return out * 0.2 + x


class LinearFourier2d(torch.nn.Module):
    def __init__(self, image_size, log):
        super(LinearFourier2d, self).__init__()

        self.log = log

        c, h, w = image_size
        self.register_parameter(name='fourier_filter', param=torch.nn.Parameter(torch.empty(c, h, w // 2 + 1)))
        torch.nn.init.ones_(self.fourier_filter)

    def forward(self, x):
        w = torch.nn.ReLU()(self.fourier_filter.repeat(x.shape[0], 1, 1, 1).to(x.device))

        rft_x = torch.rfft(x, signal_ndim=3, normalized=True, onesided=True)
        init_spectrum = torch.sqrt(torch.pow(rft_x[..., 0], 2) + torch.pow(rft_x[..., 1], 2))

        if self.log:
            spectrum = torch.exp(w * torch.log(1 + init_spectrum)) - 1
        else:
            spectrum = w * init_spectrum

        irf = torch.irfft(torch.stack([rft_x[..., 0] * spectrum / (init_spectrum + 1e-16),
                                       rft_x[..., 1] * spectrum / (init_spectrum + 1e-16)], dim=-1),
                          signal_ndim=3, normalized=True, onesided=True, signal_sizes=x.shape[1:])

        return irf


class GeneralFourier2d(torch.nn.Module):
    def __init__(self, image_size, log):
        super(GeneralFourier2d, self).__init__()

        self.log = log

        c, h, w = image_size
        self.register_parameter(name='W1', param=torch.nn.Parameter(torch.empty(c, h, w // 2 + 1)))

        # self.W1 = torch.Tensor(np.load('UNet_weights.npy')).squeeze(0)
        # self.W1 = torch.Tensor(np.load('UNet_log_weights.npy')).squeeze(0)

        # self.W1 = torch.Tensor(np.load('DenseNetSegmentation_weights.npy')).squeeze(0)
        # self.W1 = torch.Tensor(np.load('DenseNetSegmentation_log_weights.npy')).squeeze(0)

        # self.W1 = torch.Tensor(np.load('ResNetSegmentation_weights.npy')).squeeze(0)
        # self.W1 = torch.Tensor(np.load('ResNetSegmentation_log_weights.npy')).squeeze(0)

        self.register_parameter(name='B1', param=torch.nn.Parameter(torch.empty(c, h, w // 2 + 1)))
        self.register_parameter(name='W2', param=torch.nn.Parameter(torch.empty(c, h, w // 2 + 1)))
        self.register_parameter(name='B2', param=torch.nn.Parameter(torch.empty(c, h, w // 2 + 1)))

        torch.nn.init.ones_(self.W1)
        torch.nn.init.zeros_(self.B1)
        torch.nn.init.ones_(self.W2)
        torch.nn.init.zeros_(self.B2)

        # activation functions
        # self.activation = torch.nn.Sigmoid()
        self.activation = torch.nn.ReLU()
        # self.activation = torch.nn.ReLU6()
        # self.activation = torch.nn.Softplus()
        # self.activation = torch.nn.Tanh()
        # self.activation = lambda x: x * torch.nn.Sigmoid()(x)  # Swish (beta = 1.0)
        # self.activation = lambda x: x * torch.nn.Tanh()(torch.nn.Softplus()(x))  # Mish

    def forward(self, x):
        w1 = torch.nn.ReLU()(self.W1.repeat(x.shape[0], 1, 1, 1).to(x.device))
        w2 = torch.nn.ReLU()(self.W2.repeat(x.shape[0], 1, 1, 1).to(x.device))
        b1 = torch.nn.ReLU()(self.B1.repeat(x.shape[0], 1, 1, 1).to(x.device))
        b2 = torch.nn.ReLU()(self.B2.repeat(x.shape[0], 1, 1, 1).to(x.device))

        rft_x = torch.rfft(x, signal_ndim=3, normalized=True, onesided=True)
        init_spectrum = torch.sqrt(torch.pow(rft_x[..., 0], 2) + torch.pow(rft_x[..., 1], 2))

        if self.log:
            spectrum = w2 * self.activation(w1 * torch.log(1 + init_spectrum) + b1) + b2
        else:
            spectrum = w2 * self.activation(w1 * init_spectrum + b1) + b2

        irf = torch.irfft(torch.stack([rft_x[..., 0] * spectrum / (init_spectrum + 1e-16),
                                       rft_x[..., 1] * spectrum / (init_spectrum + 1e-16)], dim=-1),
                          signal_ndim=3, normalized=True, onesided=True, signal_sizes=x.shape[1:])

        return irf
