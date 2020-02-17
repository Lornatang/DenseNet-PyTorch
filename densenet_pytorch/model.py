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

from .utils import bn_function_factory
from .utils import densenet_params
from .utils import get_model_params
from .utils import load_pretrained_weights


class DenseLayer(nn.Sequential):
  def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
    super(DenseLayer, self).__init__()
    self.add_module("norm1", nn.BatchNorm2d(num_input_features)),
    self.add_module("relu1", nn.ReLU(inplace=True)),
    self.add_module("conv1", nn.Conv2d(num_input_features, bn_size *
                                       growth_rate, kernel_size=1, stride=1,
                                       bias=False)),
    self.add_module("norm2", nn.BatchNorm2d(bn_size * growth_rate)),
    self.add_module("relu2", nn.ReLU(inplace=True)),
    self.add_module("conv2", nn.Conv2d(bn_size * growth_rate, growth_rate,
                                       kernel_size=3, stride=1, padding=1,
                                       bias=False)),
    self.drop_rate = drop_rate
    self.memory_efficient = memory_efficient

  def forward(self, *prev_features):
    bn_function = bn_function_factory(self.norm1, self.relu1, self.conv1)
    if self.memory_efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
      bottleneck_output = cp.checkpoint(bn_function, *prev_features)
    else:
      bottleneck_output = bn_function(*prev_features)
    new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    if self.drop_rate > 0:
      new_features = F.dropout(new_features, p=self.drop_rate,
                               training=self.training)
    return new_features


class DenseBlock(nn.Module):
  def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False):
    super(DenseBlock, self).__init__()
    for i in range(num_layers):
      layer = DenseLayer(
        num_input_features + i * growth_rate,
        growth_rate=growth_rate,
        bn_size=bn_size,
        drop_rate=drop_rate,
        memory_efficient=memory_efficient,
      )
      self.add_module(f"denselayer{i + 1}", layer)

  def forward(self, init_features):
    features = [init_features]
    for name, layer in self.named_children():
      new_features = layer(*features)
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


class DenseNet(nn.Module):
  """Densenet-BC model class, based on
  `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
  """

  def __init__(self, blocks_args=None, global_params=None):
    # growth_rate=32, block_config=(6, 12, 24, 16),
    # num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000, memory_efficient=False):

    super().__init__()
    assert isinstance(blocks_args, list), "blocks_args should be a list"
    assert len(blocks_args) > 0, "block args must be greater than 0"
    self._global_params = global_params
    self._blocks_args = blocks_args

    # First convolution
    self.features = nn.Sequential(OrderedDict([
      ("conv0", nn.Conv2d(3, self._global_params.num_init_features, kernel_size=7, stride=2,
                          padding=3, bias=False)),
      ("norm0", nn.BatchNorm2d(self._global_params.num_init_features)),
      ("relu0", nn.ReLU(inplace=True)),
      ("pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
    ]))

    # Each denseblock
    num_features = self._global_params.num_init_features
    for i, num_layers in enumerate(self._blocks_args):
      block = DenseBlock(
        num_layers=num_layers,
        num_input_features=num_features,
        bn_size=self._global_params.bn_size,
        growth_rate=self._global_params.growth_rate,
        drop_rate=self._global_params.drop_rate,
        memory_efficient=self._global_params.memory_efficient
      )
      self.features.add_module(f"denseblock{i + 1}", block)
      num_features = num_features + num_layers * self._global_params.growth_rate
      if i != len(self._blocks_args) - 1:
        trans = Transition(num_input_features=num_features,
                           num_output_features=num_features // 2)
        self.features.add_module(f"transition{i + 1}", trans)
        num_features = num_features // 2

    # Final batch norm
    self.features.add_module("norm5", nn.BatchNorm2d(num_features))

    # Linear layer
    self.classifier = nn.Linear(num_features, self._global_params.num_classes)

    # Official init from torch repo.
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
          nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)

  def forward(self, x):
    features = self.features(x)
    out = F.relu(features, inplace=True)
    out = F.adaptive_avg_pool2d(out, (1, 1))
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
