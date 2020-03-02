# Soft Decision Tree algorithm with visualization
Author of algorithm implementation: [AaronX121](https://github.com/AaronX121/Soft-Decision-Tree)


### Introduction 
This is the pytorch implementation on Soft Decision Tree (SDT), appearing in the paper "Distilling a Neural Network Into a Soft Decision Tree". 2017 (https://arxiv.org/abs/1711.09784).

### Quick Start 
Here I offer one demo on MNIST. To run the demo, simply type the following command:
``` 
python main.py <model_path> [--load]
``` 
where  
`model_path` specifies to where .pth model file of tree will be located and save to/loaded from  
`--load` tells program to load model from .pth file instead of saving it there

### Introduction
Please see `main.py` for details on how to use SDT. For personal use, one need to change following variables:

- `use_cuda`: Actually, using CPU can be faster because you know......it's a tree
- `learner_args`: Parameters on SDT model, optimizer, etc.
- `data_dir`, `train_loader`, `test_loader`

Furthermore, main arguments in `learner_args` are:

- `depth`: tree depth (root node is with depth 0)
- `lamda`: regularization coefficient defined in equation (5) from raw paper, which decayes exponentially with the depth
- `lr`, `weight_decay`: learning rate, weight decay in optimizer

If you are interested in implementations on SDT, please see `SDT.py` for details. Instead of formally defining the structure of inner node and leaf node, I directly use one linear layer with sigmoid activation to simulate all inner nodes, for the sake of acceleration.

### Visualization
Run following command:
```
python visualize.py <model_path>
```
where  
`model_path` specifies to where .pth model file of tree created with `main.py` is located 


### Package Dependencies
SDT is developed in `python 3.6.5`. Following are the name and version of packages used in SDT: 

 - pytorch 0.4.1
 - torchvision 0.2.1
