import numpy as np
import argparse
import os
import csv
from explain import plot_to_file

parser = argparse.ArgumentParser()
parser.add_argument('fname')
parser.add_argument('--stats', action='store_true')
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
        class_masks[filename.split('.')[0]] = (1 - np.load(os.path.join(absdir, filename))).mean(axis=2).mean(axis=2).mean(axis=0).sum()
    with open(os.path.join(absdir, 'stats.csv'), 'w') as file:
        writer = csv.writer(file)
        for key, value in class_masks.items():
            layer = key.split('_')[-1]
            classname = '_'.join(key.split('_')[1:-1])
            layer = int(layer)
            writer.writerow((classname, layer, value))
else:
    for filename in files:
        class_masks = np.load(filename)
        # print(class_masks.shape)
        _class_masks = np.array([np.concatenate(pic, axis=1) for pic in class_masks])
        _class_masks_reshaped = np.concatenate(_class_masks, axis=0)
        # plot_to_file(np.expand_dims(np.mean(np.mean(np.mean(class_masks, axis=2), axis=2), axis=0), axis=0), None)
        # print(np.mean(np.mean(np.mean(class_masks, axis=2), axis=2), axis=0).shape)
        # print(np.expand_dims(np.mean(np.mean(np.mean(class_masks, axis=2), axis=2), axis=0), axis=0).shape)
        # print(class_masks.mean(axis=0).shape)
        # print(class_masks.max())
        plot_to_file(_class_masks_reshaped, str(filename).split('.')[0])
        plot_to_file(reshape_to_2d(np.expand_dims(class_masks.mean(axis=0), axis=0)), str(filename).split('.')[0]+'_agg')