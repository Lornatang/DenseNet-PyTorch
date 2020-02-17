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
import collections
import re

import torch
from torch.utils import model_zoo

########################################################################
############### HELPERS FUNCTIONS FOR MODEL ARCHITECTURE ###############
########################################################################


# Parameters for the entire model (stem, all blocks, and head)
GlobalParams = collections.namedtuple('GlobalParams', [
  'growth_rate', 'num_init_features', 'bn_size', 'drop_rate',
  'memory_efficient', 'num_classes', 'image_size'])

# Parameters for an individual model block
BlockArgs = collections.namedtuple('BlockArgs', [
  'num_layers'])

# Change namedtuple defaults
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)


def bn_function_factory(norm, relu, conv):
  def bn_function(*inputs):
    concated_features = torch.cat(inputs, 1)
    bottleneck_output = conv(relu(norm(concated_features)))
    return bottleneck_output

  return bn_function


########################################################################
############## HELPERS FUNCTIONS FOR LOADING MODEL PARAMS ##############
########################################################################


def densenet_params(model_name):
  """ Map densenet_pytorch model name to parameter coefficients. """
  params_dict = {
    # Coefficients:   growth_rate, num_init_features, res
    'densenet121': (32, 64, 224),
    'densenet161': (48, 96, 224),
    'densenet169': (32, 64, 224),
    'densenet201': (32, 64, 224),
  }
  return params_dict[model_name]


def densenet(model_name, growth_rate, num_init_features, image_size=224,
             bn_size=4, drop_rate=0, num_classes=1000, memory_efficient=False):
  """ Creates a densenet_pytorch model. """

  blocks_dict = {
    'densenet121': (6, 12, 24, 16),
    'densenet161': (6, 12, 36, 24),
    'densenet169': (6, 12, 36, 32),
    'densenet201': (6, 12, 48, 32),
  }

  blocks_args = blocks_dict[model_name]

  global_params = GlobalParams(
    growth_rate=growth_rate,
    num_init_features=num_init_features,
    image_size=image_size,
    bn_size=bn_size,
    drop_rate=drop_rate,
    num_classes=num_classes,
    memory_efficient=memory_efficient,
  )

  return blocks_args, global_params


def get_model_params(model_name, override_params):
  """ Get the block args and global params for a given model """
  if model_name.startswith('densenet'):
    g, n, s = densenet_params(model_name)
    blocks_args, global_params = densenet(
      model_name=model_name, growth_rate=g, num_init_features=n, image_size=s)
  else:
    raise NotImplementedError('model name is not pre-defined: %s' % model_name)
  if override_params:
    # ValueError will be raised here if override_params has fields not included in global_params.
    global_params = global_params._replace(**override_params)
  return list(blocks_args), global_params


urls_map = {
  'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
  'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
  'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
  'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
}


def load_pretrained_weights(model, model_name, load_fc=True):
  # '.'s are no longer allowed in module names, but previous DenseLayer
  # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
  # They are also in the checkpoints in urls_map. This pattern is used
  # to find such keys.
  state_dict = model_zoo.load_url(urls_map[model_name])
  if load_fc:
    pattern = re.compile(
      r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

    for key in list(state_dict.keys()):
      res = pattern.match(key)
      if res:
        new_key = res.group(1) + res.group(2)
        state_dict[new_key] = state_dict[key]
        del state_dict[key]
    model.load_state_dict(state_dict)
  else:
    state_dict.pop('classifier.weight')
    state_dict.pop('classifier.bias')
    model.load_state_dict(state_dict, strict=False)
  print(f"Loaded pretrained weights for {model_name}.")
