"""ResNet feature extractor (DTRB), ResNet(input_channel=1, output_channel=512, [1,2,5,3]).

Проверено по чекпоинту: layer1=1 блок, layer2=2, layer3=5, layer4=3,
conv0_1 (32,1,3,3) -> grayscale вход, conv4_2 (512,512,2,2).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = self._conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = self._conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    @staticmethod
    def _conv3x3(in_planes, out_planes, stride=1):
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class ResNet(nn.Module):
    def __init__(self, input_channel, output_channel, block, layers):
        super().__init__()
        self.output_channel_block = [
            int(output_channel / 4),
            int(output_channel / 2),
            output_channel,
            output_channel,
        ]
        self.inplanes = int(output_channel / 8)

        self.conv0_1 = nn.Conv2d(input_channel, int(output_channel / 16), 3, 1, 1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(int(output_channel / 16))
        self.conv0_2 = nn.Conv2d(int(output_channel / 16), self.inplanes, 3, 1, 1, bias=False)
        self.bn0_2 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer1 = self._make_layer(block, self.output_channel_block[0], layers[0])
        self.conv1 = nn.Conv2d(
            self.output_channel_block[0], self.output_channel_block[0], 3, 1, 1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.output_channel_block[0])

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer2 = self._make_layer(block, self.output_channel_block[1], layers[1], stride=1)
        self.conv2 = nn.Conv2d(
            self.output_channel_block[1], self.output_channel_block[1], 3, 1, 1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(self.output_channel_block[1])

        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1))
        self.layer3 = self._make_layer(block, self.output_channel_block[2], layers[2], stride=1)
        self.conv3 = nn.Conv2d(
            self.output_channel_block[2], self.output_channel_block[2], 3, 1, 1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.output_channel_block[2])

        self.layer4 = self._make_layer(block, self.output_channel_block[3], layers[3], stride=1)
        self.conv4_1 = nn.Conv2d(
            self.output_channel_block[3],
            self.output_channel_block[3],
            kernel_size=2,
            stride=(2, 1),
            padding=(0, 1),
            bias=False,
        )
        self.bn4_1 = nn.BatchNorm2d(self.output_channel_block[3])
        self.conv4_2 = nn.Conv2d(
            self.output_channel_block[3],
            self.output_channel_block[3],
            kernel_size=2,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn4_2 = nn.BatchNorm2d(self.output_channel_block[3])

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn0_1(self.conv0_1(x)))
        x = self.relu(self.bn0_2(self.conv0_2(x)))

        x = self.maxpool1(x)
        x = self.layer1(x)
        x = self.relu(self.bn1(self.conv1(x)))

        x = self.maxpool2(x)
        x = self.layer2(x)
        x = self.relu(self.bn2(self.conv2(x)))

        x = self.maxpool3(x)
        x = self.layer3(x)
        x = self.relu(self.bn3(self.conv3(x)))

        x = self.layer4(x)
        x = self.relu(self.bn4_1(self.conv4_1(x)))
        x = self.relu(self.bn4_2(self.conv4_2(x)))
        return x


class ResNet_FeatureExtractor(nn.Module):
    def __init__(self, input_channel: int = 1, output_channel: int = 512):
        super().__init__()
        self.ConvNet = ResNet(input_channel, output_channel, BasicBlock, [1, 2, 5, 3])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ConvNet(x)
