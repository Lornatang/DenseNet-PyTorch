# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from collections import OrderedDict
from typing import Any, List, Tuple

import torch
from torch import Tensor
from torch import nn

__all__ = [
    "DenseNet",
    "densenet121", "densenet161", "densenet121", "densenet201",
]


class DenseNet(nn.Module):

    def __init__(
            self,
            block_cfg: Tuple[int, int, int, int] = (6, 12, 24, 16),
            channels: int = 64,
            growth_rate: int = 32,
            bottle_neck_size: int = 4,
            dropout_rate: float = 0.0,
            num_classes: int = 1000,
    ) -> None:
        super(DenseNet, self).__init__()
        self.features = nn.Sequential(
            OrderedDict(
                [
                    ("conv0", nn.Conv2d(3, channels, (7, 7), (2, 2), (3, 3), bias=False)),
                    ("norm0", nn.BatchNorm2d(channels)),
                    ("relu0", nn.ReLU(True)),
                    ("pool0", nn.MaxPool2d((3, 3), (2, 2), (1, 1))),
                ]
            )
        )

        for i, repeat_times in enumerate(block_cfg):
            block = _DenseBlock(
                repeat_times=repeat_times,
                channels=channels,
                growth_rate=growth_rate,
                bottle_neck_size=bottle_neck_size,
                dropout_rate=dropout_rate,
            )
            self.features.add_module(f"denseblock{i + 1}", block)
            channels = channels + int(repeat_times * growth_rate)
            if i != len(block_cfg) - 1:
                trans = _Transition(channels, channels // 2)
                self.features.add_module(f"transition{i + 1}", trans)
                channels = channels // 2

        self.features.add_module("norm5", nn.BatchNorm2d(channels))
        self.features.add_module("relu5", nn.ReLU(True))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Linear(channels, num_classes)

        # Initialize neural network weights
        self._initialize_weights()

    def forward(self, x: Tensor) -> Tensor:
        out = self._forward_impl(x)

        return out

    # Support torch.script function
    def _forward_impl(self, x: Tensor) -> Tensor:
        out = self.features(x)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.constant_(module.bias, 0)


class _DenseLayer(nn.Module):
    def __init__(
            self,
            channels: int,
            growth_rate: int,
            bottle_neck_size: int,
            dropout_rate: float,
    ) -> None:
        super(_DenseLayer, self).__init__()
        growth_channels = int(bottle_neck_size * growth_rate)
        self.dropout_rate = float(dropout_rate)

        self.norm1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU(True)
        self.conv1 = nn.Conv2d(channels, growth_channels, (1, 1), (1, 1), (0, 0), bias=False)

        self.norm2 = nn.BatchNorm2d(growth_channels)
        self.relu2 = nn.ReLU(True)
        self.conv2 = nn.Conv2d(growth_channels, growth_rate, (3, 3), (1, 1), (1, 1), bias=False)
        self.dropout = nn.Dropout(dropout_rate, True)

    def forward(self, x: List[Tensor] | Tensor) -> Tensor:
        if isinstance(x, Tensor):
            x = [x]
        else:
            x = x

        out = torch.cat(x, 1)
        out = self.norm1(out)
        out = self.relu1(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        if self.dropout_rate > 0:
            out = self.dropout(out)

        return out


class _DenseBlock(nn.ModuleDict):
    def __init__(
            self,
            repeat_times: int,
            channels: int,
            growth_rate: int,
            bottle_neck_size: int,
            dropout_rate: float,
    ) -> None:
        super(_DenseBlock, self).__init__()
        for i in range(repeat_times):
            layer = _DenseLayer(
                channels=channels + i * growth_rate,
                growth_rate=growth_rate,
                bottle_neck_size=bottle_neck_size,
                dropout_rate=dropout_rate,
            )
            self.add_module(f"denselayer{i + 1}", layer)

    def forward(self, x: List[Tensor] | Tensor) -> Tensor:
        out = [x]

        for _, layer in self.items():
            denselayer_out = layer(out)
            out.append(denselayer_out)
        out = torch.cat(out, 1)

        return out


class _Transition(nn.Sequential):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
    ) -> None:
        super(_Transition, self).__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(True)
        self.conv = nn.Conv2d(in_channels, out_channels, (1, 1), (1, 1), (0, 0), bias=False)
        self.pool = nn.AvgPool2d((2, 2), (2, 2))


def densenet121(**kwargs: Any) -> DenseNet:
    model = DenseNet((6, 12, 24, 16), 64, 32, **kwargs)

    return model


def densenet161(**kwargs: Any) -> DenseNet:
    model = DenseNet((6, 12, 36, 24), 96, 48, **kwargs)

    return model


def densenet169(**kwargs: Any) -> DenseNet:
    model = DenseNet((6, 12, 32, 32), 64, 32, **kwargs)

    return model


def densenet201(**kwargs: Any) -> DenseNet:
    model = DenseNet((6, 12, 48, 32), 64, 32, **kwargs)

    return model
