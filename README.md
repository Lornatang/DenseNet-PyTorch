# DenseNet

Now supports the more efficient DenseNet-BC (DenseNet-Bottleneck-Compressed) networks. Using the DenseNet-BC-190-40 model, it obtaines state of the art performance on CIFAR-10 and CIFAR-100.

### Update (January 15, 2020)

This update allows you to use NVIDIA's Apex tool for accelerated training. By default choice `hybrid training precision` + `dynamic loss amplified` version, if you need to learn more and details about `apex` tools, please visit https://github.com/NVIDIA/apex.

### Update (January 6, 2020)

This update adds a modular neural network, making it more flexible in use. It can be deployed to many common dataset classification tasks. Of course, it can also be used in your products.

### Overview
This repository contains an op-for-op PyTorch reimplementation of [DenseNet](https://arxiv.org/pdf/1608.06993.pdf).

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
4. [Contributing](#contributing) 

### About DenseNet

If you're new to DenseNets, here is an explanation straight from the official PyTorch implementation: 

Recent work has shown that convolutional networks can be substantially deeper, more accurate, and efficient to train if they contain shorter connections between layers close to the input and those close to the output. In this paper, we embrace this observation and introduce the Dense Convolutional Network (DenseNet), which connects each layer to every other layer in a feed-forward fashion. Whereas traditional convolutional networks with L layers have L connections - one between each layer and its subsequent layer - our network has L(L+1)/2 direct connections. For each layer, the feature-maps of all preceding layers are used as inputs, and its own feature-maps are used as inputs into all subsequent layers. DenseNets have several compelling advantages: they alleviate the vanishing-gradient problem, strengthen feature propagation, encourage feature reuse, and substantially reduce the number of parameters. We evaluate our proposed architecture on four highly competitive object recognition benchmark tasks (CIFAR-10, CIFAR-100, SVHN, and ImageNet). DenseNets obtain significant improvements over the state-of-the-art on most of them, whilst requiring less memory and computation to achieve high performance.

### Installation

Install from source:
```bash
git clone https://github.com/lornatang/DenseNet
cd DenseNet
python setup.py install
``` 

### Usage

#### Loading pretrained models

Load an densenet121 network:
```python
from densenet import DenseNet
model = DenseNet.from_name("densenet121")
```

Load a pretrained densenet11: 
```python
from densenet import DenseNet
model = DenseNet.from_pretrained("densenet121")
```

Details about the models are below: 

![result.png](https://github.com/titu1994/DenseNet/raw/master/images/accuracy_densenet.JPG?raw=true)


#### Example: Classification

We assume that in your current directory, there is a `img.jpg` file and a `labels_map.txt` file (ImageNet class names). These are both included in `examples/simple`. 

```python
import json
from PIL import Image
import torch
from torchvision import transforms

from densenet import DenseNet
model = DenseNet.from_pretrained("densenet121")

# Preprocess image
tfms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
img = tfms(Image.open('img.jpg')).unsqueeze(0)
print(img.shape) # torch.Size([1, 3, 224, 224])

# Load ImageNet class names
labels_map = json.load(open('labels_map.txt'))
labels_map = [labels_map[str(i)] for i in range(1000)]

# Classify
model.eval()
with torch.no_grad():
    outputs = model(img)

# Print predictions
print('-----')
for idx in torch.topk(outputs, k=5).indices.squeeze(0).tolist():
    prob = torch.softmax(outputs, dim=1)[0, idx].item()
    print('{label:<75} ({p:.2f}%)'.format(label=labels_map[idx], p=prob*100))
```

#### ImageNet

See `examples/imagenet` for details about evaluating on ImageNet.

### Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions, simply post them as GitHub issues.   

I look forward to seeing what the community does with these models! 