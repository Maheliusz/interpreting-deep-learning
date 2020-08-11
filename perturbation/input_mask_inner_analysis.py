import numpy as np
import os 
import argparse
import csv

parser = argparse.ArgumentParser()
parser.add_argument('--path', required=False, default='.')
args = parser.parse_args()

all_files = list(filter(lambda filename: filename.endswith('.npy'), os.listdir(args.path)))
files = list(filter(lambda filename: 'modified' in filename or 'original' in filename, all_files))

with open(os.path.join(args.path, 'input_mask_inner_analysis.csv'), 'w') as f:
    writer = csv.DictWriter(f, fieldnames=('name', 'layer', 'masked', 'var', 'max', 'min'))
    writer.writeheader()
    for i, filename in enumerate(files):  
        arr = np.load(os.path.join(args.path, filename))
        itemname = filename.split('.')[0].split('_')[0]
        masked = filename.split('.')[0].split('_')[-1] == 'modified'
        layer = '_'.join(filename.split('.')[0].split('_')[1:-1])
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
