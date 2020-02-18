# DenseNet-PyTorch

`Note: Now supports the more efficient DenseNet-BC (DenseNet-Bottleneck-Compressed) networks. Using the DenseNet-BC-190-40 model, it obtaines state of the art performance on CIFAR-10 and CIFAR-100.`

### Update (Feb 18, 2020)

The update is for ease of use and deployment.

 * [Example: Export to ONNX](#example-export-to-onnx)
 * [Example: Extract features](#example-feature-extraction)
 * [Example: Visual](#example-visual)

It is also now incredibly simple to load a pretrained model with a new number of classes for transfer learning:

```python
from densenet_pytorch import DenseNet 
model = DenseNet.from_pretrained('densenet121', num_classes=10)
```

### Update (January 15, 2020)

This update allows you to use NVIDIA's Apex tool for accelerated training. By default choice `hybrid training precision` + `dynamic loss amplified` version, if you need to learn more and details about `apex` tools, please visit https://github.com/NVIDIA/apex.

### Update (January 6, 2020)

This update adds a modular neural network, making it more flexible in use. It can be deployed to many common dataset classification tasks. Of course, it can also be used in your products.

### Overview
This repository contains an op-for-op PyTorch reimplementation of [Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993.pdf).

The goal of this implementation is to be simple, highly extensible, and easy to integrate into your own projects. This implementation is a work in progress -- new features are currently being implemented.  

At the moment, you can easily:  
 * Load pretrained DenseNet models 
 * Use DenseNet models for classification or feature extraction 

_Upcoming features_: In the next few days, you will be able to:
 * Quickly finetune an DenseNet on your own dataset
 * Export DenseNet models for production
 
### Table of contents
1. [About DenseNet](#about-densenet)
2. [Installation](#installation)
3. [Usage](#usage)
    * [Load pretrained models](#loading-pretrained-models)
    * [Example: Classify](#example-classification)
    * [Example: Extract features](#example-feature-extraction)
    * [Example: Export to ONNX](#example-export-to-onnx)
    * [Example: Visual](#example-visual)
4. [Contributing](#contributing) 

### About DenseNet

If you're new to DenseNets, here is an explanation straight from the official PyTorch implementation: 

Dense Convolutional Network (DenseNet), connects each layer to every other layer in a feed-forward fashion. Whereas traditional convolutional networks with L layers have L connections - one between each layer and its subsequent layer - our network has L(L+1)/2 direct connections. For each layer, the feature-maps of all preceding layers are used as inputs, and its own feature-maps are used as inputs into all subsequent layers. DenseNets have several compelling advantages: they alleviate the vanishing-gradient problem, strengthen feature propagation, encourage feature reuse, and substantially reduce the number of parameters.

### Installation

Install from pypi:
```bash
pip install densenet_pytorch
```

Install from source:
```bash
git clone https://github.com/Lornatang/DenseNet-PyTorch
cd DenseNet-PyTorch
pip install -e .
``` 

### Usage

#### Loading pretrained models

Load an densenet121 network:
```python
from densenet_pytorch import DenseNet
model = DenseNet.from_name("densenet121")
```

Load a pretrained densenet11: 
```python
from densenet_pytorch import DenseNet
model = DenseNet.from_pretrained("densenet121")
```

Their 1-crop error rates on imagenet dataset with pretrained models are listed below.

| Model structure | Top-1 error | Top-5 error |
| --------------- | ----------- | ----------- |
|  densenet121    | 25.35       | 7.83        |
|  densenet169    | 24.00       | 7.00        |
|  densenet201    | 22.80       | 6.43        |
|  densenet161    | 22.35       | 6.20        |

#### Example: Classification

We assume that in your current directory, there is a `img.jpg` file and a `labels_map.txt` file (ImageNet class names). These are both included in `examples/simple`. 

All pre-trained models expect input images normalized in the same way,
i.e. mini-batches of 3-channel RGB images of shape `(3 x H x W)`, where `H` and `W` are expected to be at least `224`.
The images have to be loaded in to a range of `[0, 1]` and then normalized using `mean = [0.485, 0.456, 0.406]`
and `std = [0.229, 0.224, 0.225]`.

Here's a sample execution.

```python
import json

import torch
import torchvision.transforms as transforms
from PIL import Image

from densenet_pytorch import DenseNet 

# Open image
input_image = Image.open("img.jpg")

# Preprocess image
preprocess = transforms.Compose([
  transforms.Resize(256),
  transforms.CenterCrop(224),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

# Load class names
labels_map = json.load(open("labels_map.txt"))
labels_map = [labels_map[str(i)] for i in range(1000)]

# Classify with DenseNet121
model = DenseNet.from_pretrained("densenet121")
model.eval()

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
  input_batch = input_batch.to("cuda")
  model.to("cuda")

with torch.no_grad():
  logits = model(input_batch)
preds = torch.topk(logits, k=5).indices.squeeze(0).tolist()

print("-----")
for idx in preds:
  label = labels_map[idx]
  prob = torch.softmax(logits, dim=1)[0, idx].item()
  print(f"{label:<75} ({prob * 100:.2f}%)")
```

#### Example: Feature Extraction 

You can easily extract features with `model.extract_features`:
```python
import torch
from densenet_pytorch import DenseNet 
model = DenseNet.from_pretrained('densenet121')

# ... image preprocessing as in the classification example ...
inputs = torch.randn(1, 3, 224, 224)
print(inputs.shape) # torch.Size([1, 3, 224, 224])

features = model.extract_features(inputs)
print(features.shape) # torch.Size([1, 1024, 7, 7])
```

#### Example: Export to ONNX  

Exporting to ONNX for deploying to production is now simple: 
```python
import torch 
from densenet_pytorch import DenseNet 

model = DenseNet.from_pretrained('densenet121')
dummy_input = torch.randn(16, 3, 224, 224)

torch.onnx.export(model, dummy_input, "demo.onnx", verbose=True)
```

#### Example: Visual

```text
cd $REPO$/framework
sh start.sh
```

Then open the browser and type in the browser address [http://127.0.0.1:10003/](http://127.0.0.1:10003/).

Enjoy it.

#### ImageNet

See `examples/imagenet` for details about evaluating on ImageNet.

For more datasets result. Please see `research/README.md`.

### Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions, simply post them as GitHub issues.   

I look forward to seeing what the community does with these models! 
