import torch
import torch.nn as nn
from torchvision import models
from typing import Dict, Iterable, Callable

import torch.nn.functional as F
from torch.nn import init

CH_FOLD = 1
CH_FOLD2 = 2
CH_FOLD3 = 3
CH_FOLD4 = 4


class ConvBlock(nn.Module):
    """ change channel dimension only
    """
    def __init__(self, ch_in, ch_out):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class UpConv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class UNet(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(UNet, self).__init__()

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = ConvBlock(ch_in=img_ch, ch_out=int(32 * CH_FOLD))
        self.Conv2 = ConvBlock(ch_in=int(32 * CH_FOLD), ch_out=int(64 * CH_FOLD))
        self.Conv3 = ConvBlock(ch_in=int(64 * CH_FOLD), ch_out=int(128 * CH_FOLD))
        self.Conv4 = ConvBlock(ch_in=int(128 * CH_FOLD), ch_out=int(256 * CH_FOLD))
        self.Conv5 = ConvBlock(ch_in=int(256 * CH_FOLD), ch_out=int(512 * CH_FOLD))

        self.Up5 = UpConv(ch_in=int(512 * CH_FOLD), ch_out=int(256 * CH_FOLD))
        self.Up_conv5 = ConvBlock(ch_in=int(512 * CH_FOLD), ch_out=int(256 * CH_FOLD))

        self.Up4 = UpConv(ch_in=int(256 * CH_FOLD), ch_out=int(128 * CH_FOLD))
        self.Up_conv4 = ConvBlock(ch_in=int(256 * CH_FOLD), ch_out=int(128 * CH_FOLD))

        self.Up3 = UpConv(ch_in=int(128 * CH_FOLD), ch_out=int(64 * CH_FOLD))
        self.Up_conv3 = ConvBlock(ch_in=int(128 * CH_FOLD), ch_out=int(64 * CH_FOLD))

        self.Up2 = UpConv(ch_in=int(64 * CH_FOLD), ch_out=int(32 * CH_FOLD))
        self.Up_conv2 = ConvBlock(ch_in=int(64 * CH_FOLD), ch_out=int(32 * CH_FOLD))

        self.Conv_1x1 = nn.Conv2d(int(32 * CH_FOLD), output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)  # (32, L, L)

        x2 = self.max_pool(x1)
        x2 = self.Conv2(x2)  # (64, L/2, L/2)

        x3 = self.max_pool(x2)
        x3 = self.Conv3(x3)  # (128, L/4, L/4)

        x4 = self.max_pool(x3)
        x4 = self.Conv4(x4)  # (256, L/8, L/8)

        x5 = self.max_pool(x4)
        x5 = self.Conv5(x5)  # (512, L/16, L/16)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        d1 = d1.squeeze(1)

        # return torch.transpose(d1, -1, -2) * d1
        return (torch.transpose(d1, -1, -2) + d1) / 2  # symmetric


class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, layers: Iterable[str]):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in layers}

        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        _ = self.model(x)
        return self._features


def resnet_18(img_ch, num_classes):
    model_resnet = models.resnet18(weights=None)
    inplanes = 64
    model_resnet.conv1 = nn.Conv2d(img_ch, inplanes, kernel_size=7, stride=1, padding=3, bias=False)  # retain H and W
    num_features = model_resnet.fc.in_features
    model_resnet.fc = nn.Linear(num_features, num_classes)
    return model_resnet


class UNetTransfer(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(UNetTransfer, self).__init__()

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = ConvBlock(ch_in=img_ch, ch_out=int(32 * CH_FOLD))
        self.Conv2 = ConvBlock(ch_in=int(32 * CH_FOLD), ch_out=int(64 * CH_FOLD))
        self.Conv3 = ConvBlock(ch_in=int(64 * CH_FOLD), ch_out=int(128 * CH_FOLD))
        self.Conv4 = ConvBlock(ch_in=int(128 * CH_FOLD), ch_out=int(256 * CH_FOLD))
        self.Conv5 = ConvBlock(ch_in=int(256 * CH_FOLD), ch_out=int(512 * CH_FOLD))

        self.Up5 = UpConv(ch_in=int(256 * CH_FOLD4), ch_out=int(256 * CH_FOLD))
        self.Up_conv5 = ConvBlock(ch_in=int(256 * CH_FOLD3), ch_out=int(256 * CH_FOLD))

        self.Up4 = UpConv(ch_in=int(256 * CH_FOLD), ch_out=int(128 * CH_FOLD))
        self.Up_conv4 = ConvBlock(ch_in=int(128 * CH_FOLD3), ch_out=int(128 * CH_FOLD))

        self.Up3 = UpConv(ch_in=int(128 * CH_FOLD), ch_out=int(64 * CH_FOLD))
        self.Up_conv3 = ConvBlock(ch_in=int(64 * CH_FOLD2), ch_out=int(64 * CH_FOLD))

        self.Up2 = UpConv(ch_in=int(64 * CH_FOLD), ch_out=int(32 * CH_FOLD))
        self.Up_conv2 = ConvBlock(ch_in=int(64 * CH_FOLD), ch_out=int(32 * CH_FOLD))

        self.Conv_1x1 = nn.Conv2d(int(32 * CH_FOLD), output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x, cat_d3, cat_d4, cat_d5):
        # cat_d3: (B, 128, L/4, L/4) resnet18 layer 2 output
        # cat_d4: (B, 256, L/8, L/8) resnet18 layer 3 output
        # cat_d4: (B, 512, L/16, L/16) resnet18 layer 4 output
        # encoding path
        x1 = self.Conv1(x)  # (32, L, L)

        x2 = self.max_pool(x1)
        x2 = self.Conv2(x2)  # (64, L/2, L/2)

        x3 = self.max_pool(x2)
        x3 = self.Conv3(x3)  # (128, L/4, L/4)

        x4 = self.max_pool(x3)
        x4 = self.Conv4(x4)  # (256, L/8, L/8)

        x5 = self.max_pool(x4)
        x5 = self.Conv5(x5)  # (512, L/16, L/16)
        x5 = torch.cat((x5, cat_d5), dim=1)  # (256*4, L/16, L/16)

        # decoding + concat path
        d5 = self.Up5(x5)  # (256, L/8, L/8)
        d5 = torch.cat((x4, d5, cat_d4), dim=1)  # (256*3, L/8, L/8)
        d5 = self.Up_conv5(d5)  # (256, L/8, L/8)

        d4 = self.Up4(d5)  # (128, L/4, L/4)
        d4 = torch.cat((x3, d4, cat_d3), dim=1)  # (128*3, L/4, L/4)
        d4 = self.Up_conv4(d4)  # (128, L/4, L/4)

        d3 = self.Up3(d4)  # (64, L/2, L/2)
        d3 = torch.cat((x2, d3), dim=1)  # (64*2, L/2, L/2)
        d3 = self.Up_conv3(d3)  # (64, L/2, L/2)

        d2 = self.Up2(d3)  # (32, L, L)
        d2 = torch.cat((x1, d2), dim=1)  # (64, L, L)
        d2 = self.Up_conv2(d2)  # (32, L, L)

        d1 = self.Conv_1x1(d2)  # (1, L, L)
        d1 = d1.squeeze(1)  # (L, L)

        # return torch.transpose(d1, -1, -2) * d1
        return (torch.transpose(d1, -1, -2) + d1) / 2  # symmetric
