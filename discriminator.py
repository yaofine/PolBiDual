import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils import spectral_norm

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, use_spectral=False):
        super().__init__()
        conv = nn.Conv2d(in_channels, out_channels, 4, stride, 1, bias=False)
        if use_spectral:
            conv = spectral_norm(conv)
        self.block = nn.Sequential(
            conv,
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
    def forward(self, x):
        return self.block(x)

class RGBDiscriminator(nn.Module):
    def __init__(self, dim=64, num_layers=4, use_spectral=True, use_multi_scale=False):
        super().__init__()
        first_conv = nn.Conv2d(3, dim, 4, 2, 1, bias=False)
        if use_spectral:
            first_conv = spectral_norm(first_conv)
        self.first_block = nn.Sequential(first_conv, nn.LeakyReLU(0.2, inplace=True))
        layers = []
        in_channels = dim
        for i in range(num_layers):
            out_channels = min(in_channels * 2, 512)
            layers.append(ConvBlock(in_channels, out_channels, stride=2, use_spectral=use_spectral))
            in_channels = out_channels
        self.main = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        output_linear = nn.Linear(in_channels, 1)
        if use_spectral:
            output_linear = spectral_norm(output_linear)
        self.output = output_linear
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None: nn.init.constant_(m.bias, 0)
    def forward(self, x: Tensor) -> Tensor:
        x = self.first_block(x)
        x = self.main(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return self.output(x)

class GradientDiscriminator(nn.Module):
    def __init__(self, dim=64, num_layers=4, use_spectral=True):
        super().__init__()
        first_conv = nn.Conv2d(1, dim, 4, 2, 1, bias=False)
        if use_spectral:
            first_conv = spectral_norm(first_conv)
        self.first_block = nn.Sequential(first_conv, nn.LeakyReLU(0.2, inplace=True))
        layers = []
        in_channels = dim
        for i in range(num_layers):
            out_channels = min(in_channels * 2, 512)
            layers.append(ConvBlock(in_channels, out_channels, stride=2, use_spectral=use_spectral))
            in_channels = out_channels
        self.main = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        output_linear = nn.Linear(in_channels, 1)
        if use_spectral:
            output_linear = spectral_norm(output_linear)
        self.output = output_linear
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None: nn.init.constant_(m.bias, 0)
    def forward(self, x: Tensor) -> Tensor:
        x = self.first_block(x)
        x = self.main(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return self.output(x)

Discriminator = RGBDiscriminator