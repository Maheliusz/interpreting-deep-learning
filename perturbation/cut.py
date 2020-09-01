import torch
import torchvision
from torch.autograd import Variable
from torchvision import models, transforms
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
import collections
import warnings
import re
import matplotlib.pyplot as plt
import os
import csv
from utils import manipulate_weights, load_models
from scipy.stats.mstats import trim

model, loader, preprocess_image = load_models()
with open('cutting.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=('layer', 'threshold', 'mean_basic_score', 'mean_new_score'))
    writer.writeheader()
    for lname, layer in (('conv5_4', model.features[34]), ('fc2', model.classifier[3]), ('output', model.classifier[6])):
        for thr in (0.01, 0.1, 0.3, 0.5):
            rowdict = manipulate_weights(model, layer, loader, layer.weight, preprocess_image, threshold=thr, leave_smaller=False, batches=1)
            writer.writerow({
                'layer':lname, 
                'threshold':'>{}'.format(thr), 
                'mean_basic_score':rowdict[0]['mean_basic_score'], 
                'mean_new_score':rowdict[0]['mean_new_score']
            })
        for thr in (0.99, 0.9, 0.7, 0.5):
            rowdict = manipulate_weights(model, layer, loader, layer.weight, preprocess_image, threshold=thr, leave_smaller=True, batches=1)
            writer.writerow({
                'layer':lname, 
                'threshold':'<{}'.format(thr), 
                'mean_basic_score':rowdict[0]['mean_basic_score'], 
                'mean_new_score':rowdict[0]['mean_new_score']
            })

