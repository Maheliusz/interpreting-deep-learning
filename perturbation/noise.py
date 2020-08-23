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

def plot(data, filename):
    fig = plt.figure(figsize=(data.shape[1]/plt.gcf().dpi, data.shape[0]/plt.gcf().dpi))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(data, interpolation='none')
    ax.set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0)
    plt.margins(0,0)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(filename+'.jpg')

def noisify(model, layer, loader, preprocess_image=(lambda x: x), drop=0.8, mean=0.0, std=0.01, batches=10, maxiter=10):
    original_weights = layer.weight
    to_image = transforms.ToPILImage()
    for batch, (_images, labels) in enumerate(iter(loader)):
        images = [preprocess_image(img) for img in _images]
        if batch==batches:
            break
        targets = [(torch.nn.Softmax()(model(img))).cpu().data.numpy() for img in images]
        categories = [np.argmax(target) for target in targets]
        basic_scores = [targets[i][0, categories[i]] for i in range(len(categories))]
        mean_basic_score = np.mean(basic_scores)
        # basic_score = target_np[0, basic_category]
        for i in range(maxiter):
            modified_weights = layer.weight + torch.normal(mean, std, size=layer.weight.shape)
            with torch.no_grad():
                layer.weight = torch.nn.Parameter(modified_weights)
            new_targets = [(torch.nn.Softmax()(model(img))).cpu().data.numpy() for img in images]
            new_scores = [new_targets[i][0, categories[i]] for i in range(len(categories))]
            mean_new_score = np.mean(new_scores)
            print('{}/{}\t{}/{}\t{}\t{}'.format(batch+1, batches, i+1, maxiter, mean_basic_score, mean_new_score), end='\r')
            if mean_new_score < drop * mean_basic_score or mean_new_score * drop > mean_basic_score:
                with torch.no_grad():
                    layer.weight = torch.nn.Parameter((original_weights + modified_weights) / 2)
                break
    difference = torch.abs(original_weights-layer.weight)
    if len(difference.shape)>2:
        difference = difference.mean(axis=2).mean(axis=2)
    print()
    return difference

def differentiate(model, layer, loader, difference, preprocess_image=(lambda x: x), threshold=0.99, batches=10):
    to_image = transforms.ToPILImage()
    result = []
    original_weights = layer.weight
    for batch, (_images, labels) in enumerate(iter(loader)):
        with torch.no_grad():
            layer.weight = original_weights
        images = [preprocess_image(img) for img in _images]
        if batch==batches:
            break
        targets = [(torch.nn.Softmax()(model(img))).cpu().data.numpy() for img in images]
        categories = [np.argmax(target) for target in targets]
        basic_scores = [targets[i][0, categories[i]] for i in range(len(categories))]
        mean_basic_score = np.mean(basic_scores)
        binmask = torch.from_numpy(difference < threshold)
        with torch.no_grad():
            layer.weight = torch.nn.Parameter(layer.weight.T.mul(binmask.T).T)
        new_targets = [(torch.nn.Softmax()(model(img))).cpu().data.numpy() for img in images]
        new_scores = [new_targets[i][0, categories[i]] for i in range(len(categories))]
        mean_new_score = np.mean(new_scores)
        result.append({
            'thr':threshold,
            'num':batch,
            'mean_basic_score':mean_basic_score,
            'mean_new_score':mean_new_score
        })
        print('{}/{}'.format(batch+1, batches), end='\r')
    print()
    return result


def load_models():
    model = models.vgg19(pretrained=True) 
    model.eval()
    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = False 

    def preprocess_image(img):
        means=[0.485, 0.456, 0.406]
        stds=[0.229, 0.224, 0.225]

        to_image = transforms.ToPILImage()
        # img = np.array(to_image(img).convert('RGB'))[:, :, ::-1].copy()

        preprocessed_img = np.float32(np.array(to_image(img).convert('RGB'))[:, :, ::-1].copy()) / 255
        # preprocessed_img = np.float32(img) / 255
        # preprocessed_img = img.copy()[: , :, ::-1]
        # preprocessed_img = img[: , :, ::-1]
        # preprocessed_img = (to_image(img).convert('RGB')).copy()[: , :, ::-1]
        for i in range(3):
            preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
            preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
        preprocessed_img = \
            np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))

        preprocessed_img_tensor = torch.from_numpy(preprocessed_img)

        preprocessed_img_tensor.unsqueeze_(0)
        return Variable(preprocessed_img_tensor, requires_grad = False)

    
    trainset = torchvision.datasets.CIFAR100(root='./data100', train=False,
                                    download=True, transform=transforms.ToTensor())
    loader = torch.utils.data.DataLoader(trainset, batch_size=10,
                                            shuffle=True, num_workers=0)
    return model, loader, preprocess_image                                            



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=False)
    parser.add_argument('--plot', required=False, action='store_true')
    args = parser.parse_args()

    if not args.path:
        model, loader, preprocess_image = load_models()
        # layer = model.features[34]    
        # layer = model.classifier[0] 
        for lname, layer in (('conv5_4', model.features[34]), ('fc2', model.classifier[3])): 
            for std in (0.01, 0.1, 0.5):                                           
                diff = noisify(model, layer, loader, preprocess_image=preprocess_image)
                np.save('noise-{}-{}'.format(lname, int(std*100)), diff.cpu().data.numpy())
    else:
        data = np.load(args.path)
        if args.plot:
            plot(data, args.path.split(os.path.sep)[-1].split('.')[0])
        else:
            model, loader, preprocess_image = load_models()
            if 'conv5_4' in args.path:
                layer = model.features[34]
            if 'fc2' in args.path:
                layer = model.classifier[3]
            # dictlist = differentiate(model, layer, loader, data, preprocess_image)
            with open('.'.join(args.path.split('.')[:-1])+'.csv', 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=('thr', 'num', 'mean_basic_score', 'mean_new_score'))
                writer.writeheader()
                for thr in (0.01, 0.001,):
                    writer.writerows(differentiate(model, layer, loader, data, preprocess_image, threshold=thr))
