# Copyright 2020 Lorna Authors. All Rights Reserved.
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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import Tensor
from torch.jit.annotations import List

from .utils import any_requires_grad
from .utils import densenet_params
from .utils import get_model_params
from .utils import load_pretrained_weights


class DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
        r""" Neural Network Layer for Standard Structures.

        Args:
          num_input_features (int): The number of filters to learn in the first convolution layer.
          growth_rate (int): How many filters to add each layer (`k` in paper).
          bn_size (int): Multiplicative factor for number of bottle neck layers.
            (i.e. bn_size * k features in the bottleneck layer)
          drop_rate (float): Dropout rate after each dense layer.
          memory_efficient (bool): If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
        """

        super(DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs):
        # type: (List[Tensor]) -> Tensor
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        return bottleneck_output

    @torch.jit.unused
    def call_checkpoint_bottleneck(self, x):
        # type: (List[Tensor]) -> Tensor
        def closure(*inputs):
            return self.bn_function(*inputs)

        return cp.checkpoint(closure, x)

    # torchscript does not yet support *args, so we overload method
    # allowing it to take either a List[Tensor] or single Tensor
    def forward(self, inputs):
        if isinstance(inputs, Tensor):
            prev_features = [inputs]
        else:
            prev_features = inputs

        if self.memory_efficient and any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")

            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features


class DenseBlock(nn.ModuleDict):
    def __init__(self, num_layers, num_input_features, global_params):
        r""" Block of multilayer network layer

        Args:
          num_layers (int): How many layers in each pooling block
          num_input_features (int): The number of filters to learn in the first convolution layer
          global_params (namedtuple): A set of GlobalParams shared between blocks
        """

        super(DenseBlock, self).__init__()

        for i in range(num_layers):
            layer = DenseLayer(
                num_input_features + i * global_params.growth_rate,
                growth_rate=global_params.growth_rate,
                bn_size=global_params.bn_size,
                drop_rate=global_params.drop_rate,
                memory_efficient=global_params.memory_efficient,
                )
            self.add_module(f"denselayer{i + 1}", layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(Transition, self).__init__()
        self.add_module("norm", nn.BatchNorm2d(num_input_features))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module("conv", nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module("pool", nn.AvgPool2d(kernel_size=2, stride=2))


# Densenet-BC model class, based on
# `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
class DenseNet(nn.Module):
    def __init__(self, blocks_args=None, global_params=None):
        """ An EfficientNet model. Most easily loaded with the .from_name or .from_pretrained methods

        Args:
            blocks_args (list): A list of BlockArgs to construct blocks
            global_params (namedtuple): A set of GlobalParams shared between blocks

        Examples:
            model = DenseNet.from_pretrained('densenet121')
          """

        super(DenseNet, self).__init__()
        assert isinstance(blocks_args, tuple), 'blocks_args should be a tuple'
        assert len(blocks_args) > 0, 'block args must be greater than 0'

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ("conv0", nn.Conv2d(3, global_params.num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ("norm0", nn.BatchNorm2d(global_params.num_init_features)),
            ("relu0", nn.ReLU(inplace=True)),
            ("pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = global_params.num_init_features

        for i, num_layers in enumerate(blocks_args):
            block = DenseBlock(num_layers, num_features, global_params)
            self.features.add_module(f"denseblock{i + 1}", block)
            num_features = num_features + num_layers * global_params.growth_rate

            if i != len(blocks_args) - 1:
                transition = Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module(f"transition{i + 1}", transition)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module("norm5", nn.BatchNorm2d(num_features))

        # Final linear layer
        self.classifier = nn.Linear(num_features, global_params.num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def extract_features(self, inputs):
        """ Returns output of the final convolution layer """
        x = self.features(inputs)
        return x

    def forward(self, inputs):
        features = self.features(inputs)
        features = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(features, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

    @classmethod
    def from_name(cls, model_name, override_params=None):
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        return cls(blocks_args, global_params)

    @classmethod
    def from_pretrained(cls, model_name, num_classes=1000):
        model = cls.from_name(model_name, override_params={"num_classes": num_classes})
        load_pretrained_weights(model, model_name, load_fc=(num_classes == 1000))
        return model

    @classmethod
    def get_image_size(cls, model_name):
        cls._check_model_name_is_valid(model_name)
        _, _, res = densenet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name):
        """ Validates model name. None that pretrained weights are only available for
        the first four models (densenet{i} for i in 121,161,169,201) at the moment. """
        num_models = [121, 161, 169, 201]
        valid_models = ["densenet" + str(i) for i in num_models]
        if model_name not in valid_models:
            raise ValueError("model_name should be one of: " + ", ".join(valid_models))
