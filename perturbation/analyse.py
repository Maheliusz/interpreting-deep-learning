import numpy as np
import argparse
import os
import csv
from utils import plot_to_file

parser = argparse.ArgumentParser()
parser.add_argument('fname')
parser.add_argument('--stats', action='store_true')
parser.add_argument('--filter', action='store_true')
args = parser.parse_args()

def reshape_to_2d(arr):
    return np.concatenate(np.array([np.concatenate(pic, axis=1) for pic in arr]), axis=0)


# absdir = os.path.abspath(args.fname)
absdir = args.fname
if os.path.isdir(absdir):
    files = list(map(lambda filename: os.path.join(absdir, filename), filter(lambda filename: filename.endswith('.npy'), os.listdir(absdir))))
else:
    files = [absdir]
if args.stats:
    class_masks = {}
    for filename in files:
        class_masks[filename.split('.')[0]] = (1 - np.load(os.path.join(filename))).mean(axis=2).mean(axis=2).mean(axis=0).sum()
    with open(os.path.join(absdir, 'stats.csv'), 'w') as file:
        writer = csv.writer(file)
        for key, value in class_masks.items():
            layer = key.split('_')[-1]
            classname = '_'.join(key.split('_')[1:-1])
            layer = int(layer)
            writer.writerow((classname, layer, value))
elif args.filter:
    layers = list(set(['_'.join(x.split('.')[0].split('\\')[-1].split('_')[-2:]) for x in files]))
    layers = list(set(map(lambda x: x.split('_')[-1] if 'fc' in x else x, layers)))
    for layer in layers:
        layer_masks = []
        for filename in filter(lambda x: layer in x, files):
            layer_masks.append(np.load(filename))
        layer_masks = np.array(layer_masks)
        layer_masks = np.reshape(layer_masks, (-1,)+layer_masks.shape[2:])
        np.save(layer+'_agg', layer_masks)
        if len(layer_masks.shape) > 3:
            _class_masks = layer_masks.mean(axis=0)
            _class_masks_reshaped = np.concatenate(_class_masks, axis=0).T
        else:
            _class_masks_reshaped = np.expand_dims(np.squeeze(layer_masks, axis=1).mean(axis=0), axis=0)
        plot_to_file(_class_masks_reshaped, layer+'_agg')
else:
    for filename in files:
        class_masks = np.load(filename)
        if len(class_masks.shape) > 3:
            _class_masks = class_masks.mean(axis=0)
            _class_masks_reshaped = np.concatenate(_class_masks, axis=0).T
        else:
            _class_masks_reshaped = np.expand_dims(np.squeeze(class_masks, axis=1).mean(axis=0), axis=0)
        plot_to_file(_class_masks_reshaped, str(filename).split('.')[0]+'_agg')