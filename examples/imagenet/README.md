### Imagenet

This is a preliminary directory for evaluating the model on ImageNet. It is adapted from the standard PyTorch Imagenet script. 

For now, only evaluation is supported, but I am currently building scripts to assist with training new models on Imagenet. 

Example commands: 
```bash
# Evaluate small DenseNet on CPU
python main.py data -e -a DenseNet121 --pretrained 
```
```bash
# Evaluate medium DenseNet on GPU
python main.py data -e -a DenseNet121 --pretrained --gpu 0 --batch-size 128
```
```bash
# Evaluate ResNet-50 for comparison
python main.py data -e -a resnet50 --pretrained --gpu 0
```
