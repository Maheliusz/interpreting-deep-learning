import torch
from torch.autograd import Variable
from torchvision import models, transforms, datasets
import numpy as np
import argparse
import collections
import warnings
import re
import os
import csv
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, PowerNorm

parser = argparse.ArgumentParser()
parser.add_argument('path')
args = parser.parse_args()

classes = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 
'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 
'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 
'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 
'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 
'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 
'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 
'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 
'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 
'wolf', 'woman', 'worm'] #for CIFAR100, fine names

# model = models.vgg19(pretrained=True) 
files = list(filter(lambda filename: filename.endswith('.npy'), os.listdir(args.path)))

for _class in classes[:10]:
    filtered_files = filter(lambda x: _class in x, files)
    for inp, outp in (('conv5_3', 'conv5_4'),('conv5_4', 'fc1'),('fc1', 'fc2'),('fc2', 'output')):
        inmask = torch.from_numpy(np.load(os.path.join(args.path, 'class-{}-{}.npy'.format(_class, inp))).mean(axis=0))
        outmask = torch.from_numpy(np.load(os.path.join(args.path, 'class-{}-{}.npy'.format(_class, outp))).mean(axis=0))
        if 'conv' in inp:
            inmask = inmask.mean(axis=1).mean(axis=1)
        if 'conv' in outp:
            outmask = outmask.mean(axis=1).mean(axis=1)
        corr = np.outer(inmask, outmask)
        plt.imshow(corr, norm=LogNorm())
        plt.ylabel('in')
        plt.xlabel('out')
        plt.savefig(os.path.join(args.path,'corr-{}-{}.png'.format(_class, outp)))
        

for k, filename in enumerate(files):
    print(filename)
    continue
    mask = torch.from_numpy(np.load(os.path.join(args.path, filename)).mean(axis=0))
    _, classname, layername = filename.replace('.npy', '').split('-')
    analyzed_class = classes.index(classname)
    layer = None
    if 'conv5_3' == layername:
        layer = model.features[32]
    elif 'conv5_4' == layername:
        layer = model.features[34]
    elif 'fc1' == layername:
        layer = model.classifier[0]
    print('{}\t:\t{}'.format(mask.shape, layer.weight.shape))
    