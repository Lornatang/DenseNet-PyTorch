# DenseNet-PyTorch

## Overview

This repository contains an op-for-op PyTorch reimplementation
of [Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993v5.pdf).

## Table of contents

- [DenseNet-PyTorch](#densenet-pytorch)
    - [Overview](#overview)
    - [Table of contents](#table-of-contents)
    - [Download weights](#download-weights)
    - [Download datasets](#download-datasets)
    - [How Test and Train](#how-test-and-train)
        - [Test](#test)
        - [Train model](#train-model)
        - [Resume train model](#resume-train-model)
    - [Result](#result)
    - [Contributing](#contributing)
    - [Credit](#credit)
        - [Densely Connected Convolutional Networks](#densely-connected-convolutional-networks)

## Download weights

- [Google Driver](https://drive.google.com/drive/folders/17ju2HN7Y6pyPK2CC_AqnAfTOe9_3hCQ8?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1yNs4rqIb004-NKEdKBJtYg?pwd=llot)

## Download datasets

Contains MNIST, CIFAR10&CIFAR100, TinyImageNet_200, MiniImageNet_1K, ImageNet_1K, Caltech101&Caltech256 and more etc.

- [Google Driver](https://drive.google.com/drive/folders/1f-NSpZc07Qlzhgi6EbBEI1wTkN1MxPbQ?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1arNM38vhDT7p4jKeD4sqwA?pwd=llot)

Please refer to `README.md` in the `data` directory for the method of making a dataset.

## How Test and Train

Both training and testing only need to modify the `config.py` file.

### Test

- line 29: `model_arch_name` change to `densenet121`.
- line 31: `model_mean_parameters` change to `[0.485, 0.456, 0.406]`.
- line 32: `model_std_parameters` change to `[0.229, 0.224, 0.225]`.
- line 34: `model_num_classes` change to `1000`.
- line 36: `mode` change to `test`.
- line 89: `model_weights_path` change to `./results/pretrained_models/DenseNet121-ImageNet_1K-30a6e303.pth.tar`.

```bash
python3 test.py
```

### Train model

- line 29: `model_arch_name` change to `densenet121`.
- line 31: `model_mean_parameters` change to `[0.485, 0.456, 0.406]`.
- line 32: `model_std_parameters` change to `[0.229, 0.224, 0.225]`.
- line 34: `model_num_classes` change to `1000`.
- line 36: `mode` change to `train`.
- line 50: `pretrained_model_weights_path` change to `./results/pretrained_models/DenseNet121-ImageNet_1K-30a6e303.pth.tar`.

```bash
python3 train.py
```

### Resume train model

- line 29: `model_arch_name` change to `densenet121`.
- line 31: `model_mean_parameters` change to `[0.485, 0.456, 0.406]`.
- line 32: `model_std_parameters` change to `[0.229, 0.224, 0.225]`.
- line 34: `model_num_classes` change to `1000`.
- line 36: `mode` change to `train`.
- line 53: `resume` change to `./samples/densenet121-ImageNet_1K/epoch_xxx.pth.tar`.

```bash
python3 train.py
```

## Result

Source of original paper results: [https://arxiv.org/pdf/1608.06993v5.pdf](https://arxiv.org/pdf/1608.06993v5.pdf))

In the following table, the top-x error value in `()` indicates the result of the project, and `-` indicates no test.

|    Model    |   Dataset   | Top-1 error (val) | Top-5 error (val) |
|:-----------:|:-----------:|:-----------------:|:-----------------:|
| densenet121 | ImageNet_1K | 32.6%(**32.3%**)  |   -(**12.5%**)    |
| densenet161 | ImageNet_1K | 24.8%(**24.7%**)  |    -(**7.4%**)    |
| densenet169 | ImageNet_1K | 24.8%(**24.7%**)  |    -(**7.4%**)    |
| densenet201 | ImageNet_1K | 24.8%(**24.7%**)  |    -(**7.4%**)    |

```bash
# Download `DenseNet121-ImageNet_1K-30a6e303.pth.tar` weights to `./results/pretrained_models`
# More detail see `README.md<Download weights>`
python3 ./inference.py 
```

Input:

<span align="center"><img width="224" height="224" src="figure/n01440764_36.JPEG"/></span>

Output:

```text
Build `densenet121` model successfully.
Load `densenet121` model weights `/DenseNet-PyTorch/results/pretrained_models/DenseNet121-ImageNet_1K-30a6e303.pth.tar` successfully.
tench, Tinca tinca                                                          (99.53%)
barracouta, snoek                                                           (0.35%)
armadillo                                                                   (0.04%)
croquet ball                                                                (0.01%)
bolete                                                                      (0.01%)
```

## Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions,
simply post them as GitHub issues.

I look forward to seeing what the community does with these models!

### Credit

#### Densely Connected Convolutional Networks

*Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang,
Vijay Vasudevan, Quoc V. Le, Hartwig Adam*

##### Abstract

Recent work has shown that convolutional networks can be substantially deeper, more accurate, and efficient to train if
they contain shorter connections between layers close to the input and those close to the output. In this paper, we
embrace this observation and introduce the Dense Convolutional Network (DenseNet), which connects each layer to every
other layer in a feed-forward fashion. Whereas traditional convolutional networks with L layers have L connections - one
between each layer and its subsequent layer - our network has L(L+1)/2 direct connections. For each layer, the
feature-maps of all preceding layers are used as inputs, and its own feature-maps are used as inputs into all subsequent
layers. DenseNets have several compelling advantages: they alleviate the vanishing-gradient problem, strengthen feature
propagation, encourage feature reuse, and substantially reduce the number of parameters. We evaluate our proposed
architecture on four highly competitive object recognition benchmark tasks (CIFAR-10, CIFAR-100, SVHN, and ImageNet).
DenseNets obtain significant improvements over the state-of-the-art on most of them, whilst requiring less computation
to achieve high performance. Code and pre-trained models are available at this https URL .

[[Paper]](https://arxiv.org/pdf/1608.06993v5.pdf)

```bibtex
@inproceedings{huang2017densely,
  title={Densely connected convolutional networks},
  author={Huang, Gao and Liu, Zhuang and Van Der Maaten, Laurens and Weinberger, Kilian Q},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={4700--4708},
  year={2017}
}
```