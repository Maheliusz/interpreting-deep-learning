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

#############################
blur_radius = 11
image_size = (32,32)
tv_beta = 3
learning_rate = 0.1
max_iterations = 80
l1_coeff = 0.01
tv_coeff = 0.2
with_tv = True
mask_scale = 1
batch_size = 10
#############################
l1_mask = None
vis = None
class_masks = []
#############################
mask_size = (0,0)

use_cuda = False #torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor
vgg = True

parser = argparse.ArgumentParser()
parser.add_argument('--imgpath', required=False)
parser.add_argument('--model', required=False)
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--multimask', action='store_true')

def tv_norm(input, tv_beta):
    img = input#[0, 0, :]
    # print(img.shape)
    if len(img.shape) > 1:
        row_grad = torch.mean(torch.abs((img[:-1 , :] - img[1 :, :])).pow(tv_beta))
        col_grad = torch.mean(torch.abs((img[: , :-1] - img[: , 1 :])).pow(tv_beta))
    else:
        row_grad = torch.mean(torch.abs((img[:-1] - img[1 :])).pow(tv_beta))
        col_grad = 0
    return row_grad + col_grad

def preprocess_image(img):
    means=[0.485, 0.456, 0.406]
    stds=[0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[: , :, ::-1]
    if vgg:
        for i in range(3):
            preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
            preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))

    if use_cuda:
        preprocessed_img_tensor = torch.from_numpy(preprocessed_img).cuda()
    else:
        preprocessed_img_tensor = torch.from_numpy(preprocessed_img)

    preprocessed_img_tensor.unsqueeze_(0)
    return Variable(preprocessed_img_tensor, requires_grad = False)

def numpy_to_torch(img, requires_grad = True):
    if len(img.shape) < 3:
        output = np.float32([img])
    else:
        output = np.transpose(img, (2, 0, 1))

    output = torch.from_numpy(output)
    if use_cuda:
        output = output.cuda()

    output.unsqueeze_(0)
    v = Variable(output, requires_grad = requires_grad)
    return v

def infohook(module, input, output):
    print(output.shape, end='\r')
    exit()
    return output

def hook_function(module, input, output):
    global vis
    vis = output.reshape(output.shape[-3:])
    np_vis = (vis.cpu() if not use_cuda else vis.cuda()).data.numpy()[:]
    if len(output.shape)>2:
        blurred_activations = np.expand_dims([cv2.GaussianBlur(_vis, (blur_radius, blur_radius), cv2.BORDER_DEFAULT) for _vis in np_vis], 0)
    else:
        blurred_activations = np.zeros(output.shape, dtype=np.float32)
    global l1_mask
    return output.mul(l1_mask) + torch.from_numpy(blurred_activations).mul(1-l1_mask)

def load_model(model_path):
    if not model_path:
        model = models.vgg19(pretrained=True) 
    else:
        model = torch.load(model_path)
        vgg = False
    model.eval()
    if use_cuda:
        model.cuda()
    
    if not model_path:
        for p in model.features.parameters():
            p.requires_grad = False
        for p in model.classifier.parameters():
            p.requires_grad = False

    return model

def process_single_image(model, original_img, verbose=False):
    original_img = cv2.resize(original_img, image_size)
    img = np.float32(original_img) / 255
    mask_init = np.ones((int(image_size[0]*mask_scale),int(image_size[1]*mask_scale)), dtype = np.float32)

    # inner layers mask initializations
    global l1_mask
    l1_mask = np.ones(vis.shape[-3:], dtype=np.float32)
    
    # Convert to torch variables
    img = preprocess_image(img)
    mask = numpy_to_torch(mask_init)
    l1_mask = torch.from_numpy(l1_mask)
    l1_mask = Variable(l1_mask, requires_grad = True)

    optimizer = torch.optim.Adam([l1_mask], lr=learning_rate)
    # upsample = torch.nn.UpsamplingBilinear2d(size=image_size)

    # img = upsample(img)
    target = torch.nn.Softmax()(model(img))
    category = np.argmax((target.cpu() if not use_cuda else target.cuda()).data.numpy())
    if verbose:
        print("Category with highest probability {}".format(category))
        print("Optimizing.. ")

    prev = 0.0
    for i in range(max_iterations):
        outputs = torch.nn.Softmax()(model(img))
        loss = l1_coeff*torch.mean(torch.abs(1 - l1_mask)) + outputs[0, category]
        for j in range(l1_mask.shape[0]):
            loss += tv_coeff*tv_norm(l1_mask[j], tv_beta) if with_tv else 0

        optimizer.zero_grad()
        # print(loss, end='\r')
        loss.backward()
        optimizer.step()
        loss_numpy = (loss.cpu() if not use_cuda else loss.cuda()).data.numpy()
        print('\tIteration {},\tloss: {:0.6f}'.format(i+1, loss_numpy), end='\r')

        # Optional: clamping seems to give better results
        l1_mask.data.clamp_(0, 1)
    np_vis = (vis.cpu() if not use_cuda else vis.cuda()).data.numpy()
    masks = (l1_mask.cpu() if not use_cuda else l1_mask.cuda()).data.numpy()
    '''
    fig = plt.figure(tight_layout=True)
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(masks, aspect = 1, interpolation='none')
    ax.set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0)
    plt.margins(0,0)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    fig.savefig('pics/vgg19_c6.png', dpi=fig.dpi, bbox_inches='tight', pad_inches=0)
    '''
    print()
    global class_masks
    class_masks.append(masks)

    # upsampled_mask = upsample(mask)
    # return upsampled_mask, original_img, blurred_img_numpy, loss_numpy
    return None, original_img, None, loss_numpy
    # return l1_mask, vis, vis, loss_numpy


def run(model, confs, analyzed_class, layer, layer_name, imgpath=None):
    global l1_mask, vis
    l1_mask = np.expand_dims(np.ones(mask_size), 0)
    vis = np.expand_dims(np.ones(mask_size), 0)
    hook = layer.register_forward_hook(hook_function)
    if imgpath:
        data = ((cv2.imread(imgpath, 1), classes.index(re.sub('[0-9]', '', imgpath.split('/')[-1].split('.')[0]))),)
    else:
        to_image = transforms.ToPILImage()
        # trainset = torchvision.datasets.CIFAR10(root='./data', train=False,
        #                                 download=True, transform=transforms.ToTensor())
        trainset = torchvision.datasets.CIFAR100(root='./data100', train=False,
                                        download=True, transform=transforms.ToTensor())
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, num_workers=0)
        data = []
        for images, labels in iter(trainloader):
            data.extend(list(filter(lambda x: x[1].numpy() == analyzed_class, list(zip(images, labels)))))
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
        if i == 25:
            break
        img, label = item, analyzed_class
        img = img if args.imgpath else np.array(to_image(img).convert('RGB'))[:, :, ::-1].copy()

        if classes[label] in class_counter:
            class_counter[classes[label]]+=1
        else:
            class_counter.update([classes[label]])
        title = classes[label]+str(class_counter[classes[label]])
        print("Processing {}{},\t{}/{}".format(classes[label], class_counter.get(classes[label], 0), i+1, len(items_of_class)))
        for j, conf in enumerate(confs):
            print("\tConf {}/{}".format(j+1, len(confs)))
            blur_radius, with_tv, l1_coeff, tv_coeff, max_iterations, mask_scale = conf
            upsampled_mask, original_img, blurred_img_numpy, loss = process_single_image(model, img, args.verbose)
    np.save('class-{}-{}'.format(classes[analyzed_class], layer_name), class_masks)
    hook.remove()


if __name__ == '__main__':
    args = parser.parse_args()
    if not args.verbose:
        warnings.filterwarnings("ignore")
    confs = [
        # [blur_radius, with_tv, l1_coeff, tv_coeff, max_iterations, mask_scale],
        [blur_radius, with_tv, l1_coeff, tv_coeff, max_iterations, mask_scale],
        # [3, with_tv, l1_coeff, tv_coeff, max_iterations, mask_scale],
        # [7, with_tv, l1_coeff, tv_coeff, max_iterations, mask_scale],
        # [15, with_tv, l1_coeff, tv_coeff, max_iterations, mask_scale],
        # [21, with_tv, l1_coeff, tv_coeff, max_iterations, mask_scale],
        # [25, with_tv, l1_coeff, tv_coeff, max_iterations, mask_scale],
        # [blur_radius, with_tv, l1_coeff*0.5, tv_coeff, max_iterations, mask_scale],
        # [blur_radius, with_tv, l1_coeff*0.1, tv_coeff, max_iterations, mask_scale],
        # [blur_radius, with_tv, 0.05, tv_coeff, max_iterations, mask_scale],
        # [blur_radius, with_tv, 0.1, tv_coeff, max_iterations, mask_scale],
        # [blur_radius, with_tv, 0.5, tv_coeff, max_iterations, mask_scale],
        # [blur_radius, with_tv, 1, tv_coeff, max_iterations, mask_scale],
        # [blur_radius, with_tv, l1_coeff, tv_coeff*0.5, max_iterations, mask_scale],
        # [blur_radius, with_tv, l1_coeff, tv_coeff*0.25, max_iterations, mask_scale],
        # [blur_radius, with_tv, l1_coeff, tv_coeff*0.0, max_iterations, mask_scale],
        # [blur_radius, with_tv, l1_coeff, tv_coeff*2, max_iterations, mask_scale],
        # [blur_radius, with_tv, l1_coeff, tv_coeff*5, max_iterations, mask_scale],
        # [blur_radius, with_tv, l1_coeff, tv_coeff*10, max_iterations, mask_scale],
        # [blur_radius, with_tv, l1_coeff, 0.05, max_iterations, mask_scale],
        # [blur_radius, with_tv, l1_coeff, 0.1, max_iterations, mask_scale],
        # [blur_radius, with_tv, l1_coeff, 0.5, max_iterations, mask_scale],
        # [blur_radius, with_tv, l1_coeff, tv_coeff, 100, mask_scale],
        # [blur_radius, with_tv, l1_coeff, tv_coeff, 2000, mask_scale],
        # [blur_radius, with_tv, l1_coeff, tv_coeff, max_iterations, 1],
        # [blur_radius, with_tv, l1_coeff, tv_coeff, max_iterations, 0.75],
        # [blur_radius, with_tv, l1_coeff, tv_coeff, max_iterations, 0.5],
        # [blur_radius, with_tv, l1_coeff, tv_coeff, max_iterations, 0.2],
        # [blur_radius, with_tv, l1_coeff, tv_coeff, max_iterations, 0.1],
    ]
    multimasks = (1, 0.75, 0.5, 0.2, 0.1)

    model = load_model(args.model)
    # classes = ('plane', 'car', 'bird', 'cat',
    #     'deer', 'dog', 'frog', 'horse', 'ship', 'truck')  #for CIFAR-10
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
    for i in range(10):
        for j in ((model.classifier[3], 'fc2'), (model.classifier[6], 'output')):#, (model.features[32], 'conv5_3'), (model.features[34], 'conv5_4'), (model.classifier[0], 'fc1'),):
            class_masks = []
            layer, layer_name = j
            if 'conv5' in layer_name:
                mask_size = (512,2,2)
            elif 'fc1' in layer_name or 'fc2' in layer_name:
                mask_size = (4096,)
            elif 'output' in layer_name:
                mask_size = (1000,)
            run(model, confs, i, layer, layer_name, args.imgpath)
