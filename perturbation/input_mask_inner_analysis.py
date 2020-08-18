import numpy as np
import os 
import argparse
import csv
import re

parser = argparse.ArgumentParser()
parser.add_argument('--path', required=False, default='.')
args = parser.parse_args()

all_files = list(filter(lambda filename: filename.endswith('.npy'), os.listdir(args.path)))
files = list(filter(lambda filename: '-' in filename, all_files))

with open(os.path.join(args.path, 'input_mask_inner_analysis.csv'), 'w') as f:
    writer = csv.DictWriter(f, fieldnames=('name', 'layer', 'masked', 'var', 'max', 'min'))
    writer.writeheader()
    classes = set(map(lambda filename: re.sub(r'\d+', '', filename.split('-')[0]), files))
    # print(classes)
    # exit()
    layers = set(map(lambda filename: filename.split('-')[1], files))
    for i, filename in enumerate(files):  
        arr = np.load(os.path.join(args.path, filename))
        itemname = filename.split('.')[0].split('-')[0]
        masked = filename.split('.')[0].split('-')[-1] == 'modified'
        layer = filename.split('.')[0].split('-')[1]
        var = np.var(arr)
        max = np.max(arr)
        min = np.min(arr)
        writer.writerow({
            'name':itemname,
            'layer':layer,
            'masked':masked,
            'var':var,
            'max':max,
            'min':min,
        })
        print('{}/{}'.format(i+1, len(files)), end='\r')
    print()
with open(os.path.join(args.path, 'input_mask_inner_analysis_class.csv'), 'w') as f:
    writer = csv.DictWriter(f, fieldnames=('class', 'layer', 'masked', 'var', 'max', 'min'))
    writer.writeheader()
    for masked in ('modified', 'original'):
        for classname in classes:
            for layer in layers:
                arr = np.array([np.squeeze(np.load(os.path.join(args.path, filename)), axis=0) for filename in filter(lambda x: classname == re.split('\d+', x)[0] and masked in x and layer in x, files)]).mean(axis=0)
                var = np.var(arr)
                max = np.max(arr)
                min = np.min(arr)
                writer.writerow({
                    'class':classname,
                    'layer':layer,
                    'masked':masked=='modified',
                    'var':var,
                    'max':max,
                    'min':min,
                })
with open(os.path.join(args.path, 'input_mask_inner_analysis_layer.csv'), 'w') as f:
    writer = csv.DictWriter(f, fieldnames=('layer', 'masked', 'var', 'max', 'min'))
    writer.writeheader()
    for masked in ('modified', 'original'):
        for layer in layers:
            arr = np.array([np.squeeze(np.load(os.path.join(args.path, filename)), axis=0) for filename in filter(lambda x: masked in x and layer in x, files)]).mean(axis=0)
            var = np.var(arr)
            max = np.max(arr)
            min = np.min(arr)
            writer.writerow({
                'layer':layer,
                'masked':masked=='modified',
                'var':var,
                'max':max,
                'min':min,
            })


