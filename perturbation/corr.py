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
from utils import load_models, manipulate_weights

parser = argparse.ArgumentParser()
parser.add_argument('path')
parser.add_argument('--plot', action='store_true')
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

model, loader, preprocess_image = load_models()

for _class in classes[:1]:
    filtered_files = filter(lambda x: _class in x, files)
    for inp, outp in (('conv5_3', 'conv5_4'),):#('fc1', 'fc2'),('fc2', 'output')):
        inmask = torch.from_numpy(np.load(os.path.join(args.path, 'class-{}-{}.npy'.format(_class, inp))).mean(axis=0))
        outmask = torch.from_numpy(np.load(os.path.join(args.path, 'class-{}-{}.npy'.format(_class, outp))).mean(axis=0))
        if 'conv' in inp:
            # inmask = inmask.mean(axis=1).mean(axis=1)
            pass
        else:
            inmask = np.squeeze(inmask, axis=0)
        if 'conv' in outp:
            outmask = outmask.mean(axis=1).mean(axis=1)
        # corr = np.outer(outmask, inmask)
        # print((outp, inp))
        # print(corr.shape)
        if args.plot:
            pass
        #     plt.imshow(corr, norm=LogNorm())
        #     plt.ylabel('in')
        #     plt.xlabel('out')
        #     plt.savefig(os.path.join(args.path,'corr-{}-{}.png'.format(_class, outp)))
        else:
            if inp == 'conv5_3':
                layer = model.features[32]
            elif inp == 'conv5_4':
                layer = model.features[34]
            elif inp == 'fc1':
                layer = model.classifier[0]
            elif inp == 'fc2':
                layer = model.classifier[3]
            elif inp == 'output':
                layer = model.classifier[6]
            weights = layer.weight.cpu().numpy()
            weights = np.swapaxes(weights, 0, 1)
            # for inw, inm in zip(weights, inmask):
            #     print('{}\t{}'.format(inw.shape, inm.shape))
            #     for outw in inw:
            #         mask = inm.reshape(-1)
            #         max_outw = np.sort(outw).reshape(-1)[-mask.shape[0]:]
            #         print(mask)
            #         print(max_outw)
            #         exit()
            inmask = 1 - inmask
            corr = np.nan_to_num([[np.corrcoef(outm.reshape(-1), np.sort(outw.reshape(-1))[-outm.reshape(-1).shape[0]:])[1,0] for outw, outm in zip(inw, inmask)] for inw in weights]).T
            # print(corr.shape)
            # print(np.min(corr))
            # print(np.max(corr))
            # continue
            # for inw in weights:
            #     for outw, outm in zip(inw, inmask):
            #         _outm = outm.reshape(-1)
            #         _outw = np.sort(outw.reshape(-1))[-_outm.shape[0]:]
            #         print(np.corrcoef(_outm, _outw))
            #         exit()
            # corr = 1-corr
            # print(np.min(corr))
            # print(np.max(corr))
            # exit()
            # dictlist = manipulate_weights(model, layer, loader, corr, preprocess_image, threshold=0.0)
            with open(os.path.join(args.path,'corr-{}-{}-lowest.csv'.format(_class, outp)), 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=('thr', 'num', 'mean_basic_score', 'mean_new_score'))
                writer.writeheader()
                for thr in (0.05, 0.1, 0.25, 0.4):
                    writer.writerows(manipulate_weights(model, layer, loader, corr, preprocess_image, threshold=thr, leave_smaller=False))
            with open(os.path.join(args.path,'corr-{}-{}-highest.csv'.format(_class, outp)), 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=('thr', 'num', 'mean_basic_score', 'mean_new_score'))
                writer.writeheader()
                for thr in (0.05, 0.1, 0.25, 0.4):
                    writer.writerows(manipulate_weights(model, layer, loader, corr, preprocess_image, threshold=1-thr))
            with open(os.path.join(args.path,'corr-{}-{}-mid.csv'.format(_class, outp)), 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=('thr', 'num', 'mean_basic_score', 'mean_new_score'))
                writer.writeheader()
                for thr in (0.05, 0.1, 0.25, 0.4):
                    writer.writerows(manipulate_weights(model, layer, loader, corr, preprocess_image, threshold=thr, leave_smaller='between'))
    