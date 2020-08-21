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

# 
pics_analyzed_per_class = 5
max_iter = 10
thresholds = (0.99, 0.95, 0.90, 0.75, 0.5)
# 

parser = argparse.ArgumentParser()
parser.add_argument('path')
args = parser.parse_args()

def preprocess_image(img):
    means=[0.485, 0.456, 0.406]
    stds=[0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[: , :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))

    preprocessed_img_tensor = torch.from_numpy(preprocessed_img)

    preprocessed_img_tensor.unsqueeze_(0)
    return Variable(preprocessed_img_tensor, requires_grad = False)

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

# masks = np.load(args.path)
# mask = masks.mean(axis=0)
# # mask = masks[0]
# mask = torch.from_numpy(mask)

model = models.vgg19(pretrained=True) 

for p in model.features.parameters():
    p.requires_grad = False
for p in model.classifier.parameters():
    p.requires_grad = False

threshold = 1
layername = ''
def hook_fun(module, input, output):
    binmask = mask < threshold 
    return output.mul(binmask) 
def hook_53(module, input, output):
    binmask = conv53_mask < threshold 
    return output.mul(binmask) 
def hook_54(module, input, output):
    binmask = conv54_mask < threshold 
    return output.mul(binmask) 
def hook_fc1(module, input, output):
    binmask = fc1_mask < threshold 
    return output.mul(binmask) 
def hook_fc2(module, input, output):
    binmask = fc2_mask < threshold 
    return output.mul(binmask) 
def hook_fc3(module, input, output):
    binmask = fc3_mask < threshold 
    return output.mul(binmask) 

to_image = transforms.ToPILImage()
# trainset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                 download=True, transform=transforms.ToTensor())
trainset = datasets.CIFAR100(root='./data100', train=False,
                download=True, transform=transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=10,
                        shuffle=False, num_workers=0)

files = list(filter(lambda filename: filename.endswith('.npy'), os.listdir(args.path)))
          

for t, thr in enumerate(thresholds):
    break
    threshold = thr
    f = open(os.path.join(args.path, 'threshold-{}'.format(int(threshold*100))+'.csv'), 'w', newline='')
    writer = csv.DictWriter(f, fieldnames=('item', 'masked_layer', 'basic_score', 'score_mean', 'score_var', 'score_min', 'score_max', 'wrong_classification'))
    writer.writeheader()   
    for k, filename in enumerate(files):
        # break
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
        elif 'fc2' == layername:
            layer = model.classifier[3]
        elif 'output' == layername:
            layer = model.classifier[6]
            # continue
        data = []
        for images, labels in iter(trainloader):
            data.extend(list(filter(lambda x: x[1].numpy() == analyzed_class, list(zip(images, labels)))))
            # data.extend(list(zip(images, labels))
        class_counter = collections.Counter()
        class_dict = {}
        for i in data:
            idx = int(i[1].numpy())
            if idx in class_dict.keys():
                class_dict[idx].append(i[0])
            else:
                class_dict[idx] = [i[0]]
            items_of_class = class_dict[analyzed_class]   
        for i, item in enumerate(items_of_class):
            if i == pics_analyzed_per_class:
                break
            img, label = item, analyzed_class
            img = np.array(to_image(img).convert('RGB'))[:, :, ::-1].copy()
            img = preprocess_image(np.float32(img) / 255)

            target = torch.nn.Softmax()(model(img))
            target_np = target.cpu().data.numpy()
            basic_category = np.argmax(target_np)     
            basic_score = target_np[0, basic_category]
            wrong_counter = 0
            scores = []
            hook = layer.register_forward_hook(hook_fun)
            # print('{:.3f}\t{:.3f}'.format(torch.min(mask), torch.max(mask)))
            for j in range(max_iter):
                # print('Var:{:.3f}\tmin:{:.3f}\tmax:{:.3f}\tcat:{}\t{}/{}'.format(np.var(target_np),np.min(target_np),np.max(target_np), category, i+1, max_iter))
                target = torch.nn.Softmax()(model(img))
                target_np = target.cpu().data.numpy()
                category = np.argmax(target_np)
                score = target_np[0, basic_category]
                if category != basic_category:
                    wrong_counter+=1
                # print('Var:{:.3f}\tmin:{:.3f}\tmax:{:.3f}\tcat:{}\toldmax:{:.3f}\t{}/{}'.format(np.var(target_np),np.min(target_np),np.max(target_np), category, score, i+1, max_iter))
                print('{:2d}/{:2d}\t{:2d}/{:2d}\t{:2d}/{:2d}\t{:2d}/{:2d}'.format(t+1, len(thresholds), k+1, len(files), i+1, pics_analyzed_per_class, j+1, max_iter), end='\r')
                # exit()
                scores.append(score)
            writer.writerow({
                'item': '{}{}'.format(classname, i+1), 
                'masked_layer': layername, 
                'basic_score': basic_score, 
                'score_mean': np.mean(scores), 
                'score_var': np.var(scores), 
                'score_min': np.min(scores), 
                'score_max': np.max(scores), 
                'wrong_classification': wrong_counter/max_iter
            })
            hook.remove()

print()
for t, thr in enumerate(thresholds):
    threshold = thr
    for i, _class in enumerate(classes[:1]):  
        conv53_mask = torch.from_numpy(np.load(os.path.join(args.path, 'class-{}-conv5_3.npy'.format(_class))).mean(axis=0))
        conv54_mask = torch.from_numpy(np.load(os.path.join(args.path, 'class-{}-conv5_4.npy'.format(_class))).mean(axis=0))
        fc1_mask = torch.from_numpy(np.load(os.path.join(args.path, 'class-{}-fc1.npy'.format(_class))).mean(axis=0))
        fc2_mask = torch.from_numpy(np.load(os.path.join(args.path, 'class-{}-fc2.npy'.format(_class))).mean(axis=0))
        fc3_mask = torch.from_numpy(np.load(os.path.join(args.path, 'class-{}-output.npy'.format(_class))).mean(axis=0))
        f = open(os.path.join(args.path, 'threshold-{}-{}'.format(int(threshold*100), _class)+'.csv'), 'w', newline='')
        writer = csv.DictWriter(f, fieldnames=('class', 'score_mean', 'score_var', 'masked_score_mean', 'masked_score_var'))
        writer.writeheader()

        for j, _class2 in enumerate(classes[:10]):  
            analyzed_class = classes.index(_class2)
            data = []
            for images, labels in iter(trainloader):
                data.extend(list(filter(lambda x: x[1].numpy() == analyzed_class, list(zip(images, labels)))))
                # data.extend(list(zip(images, labels))
            class_counter = collections.Counter()
            class_dict = {}
            for d in data:
                idx = int(d[1].numpy())
                if idx in class_dict.keys():
                    class_dict[idx].append(d[0])
                else:
                    class_dict[idx] = [d[0]]
                items_of_class = class_dict[analyzed_class]  
            basic_scores = []
            after_scores = []
            for k, item in enumerate(items_of_class):
                if k == pics_analyzed_per_class:
                    break
                img, label = item, analyzed_class
                img = np.array(to_image(img).convert('RGB'))[:, :, ::-1].copy()
                img = preprocess_image(np.float32(img) / 255)
                target = (torch.nn.Softmax()(model(img))).cpu().data.numpy()
                basic_category = np.argmax(target)     
                basic_scores.append(target[0, basic_category])
                h1 = model.features[32].register_forward_hook(hook_53)
                h2 = model.features[34].register_forward_hook(hook_54)
                h3 = model.classifier[0].register_forward_hook(hook_fc1)
                h4 = model.classifier[3].register_forward_hook(hook_fc2)
                h5 = model.classifier[6].register_forward_hook(hook_fc3)
                target = (torch.nn.Softmax()(model(img))).cpu().data.numpy()
                after_scores.append(target[0, basic_category])
                h1.remove()
                h2.remove()
                h3.remove()
                h4.remove()
                h5.remove()
                print('{}/{}\t{}/{}\t{}/{}\t{}/{}'.format(t+1, len(thresholds), i+1, len(classes[:3]), j+1, len(classes[:3]), k+1, pics_analyzed_per_class), end='\r')
            writer.writerow({
                'class': _class2,
                'score_mean': np.mean(basic_scores), 
                'score_var': np.var(basic_scores),
                'masked_score_mean': np.mean(after_scores), 
                'masked_score_var': np.var(after_scores)
            })
            # print()
    # print('{}/{}'.format(t+1, len(thresholds)))
            
