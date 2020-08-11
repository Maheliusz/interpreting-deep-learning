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

#############################
blur_radius = 11
image_size = (32,32)
tv_beta = 3
learning_rate = 0.1
max_iterations = 100
l1_coeff = 0.01
tv_coeff = 0.2
with_tv = True
mask_scale = 1
batch_size = 10
#############################

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor
vgg = True
state = 'original'
layer_name = ''
item_name = ''

parser = argparse.ArgumentParser()
parser.add_argument('--imgpath', required=False)
parser.add_argument('--model', required=False)
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--multimask', action='store_true')

def tv_norm(input, tv_beta):
    img = input[0, 0, :]
    row_grad = torch.mean(torch.abs((img[:-1 , :] - img[1 :, :])).pow(tv_beta))
    col_grad = torch.mean(torch.abs((img[: , :-1] - img[: , 1 :])).pow(tv_beta))
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

def save(masks, img, blurs, filename, loss):
    mask = None
    for item in masks:
        _mask = item.cpu().data.numpy()[0]
        mask = _mask if mask is None else np.add(mask, _mask)
    mask /= len(masks)
    # mask = mask.cpu().data.numpy()[0]
    mask = np.transpose(mask, (1, 2, 0))
    blurred = None
    for item in blurs:
        blurred = item if blurred is None else np.add(blurred, item)
    blurred /= len(blurs)

    mask = (mask - np.min(mask)) / np.max(mask)
    mask = 1 - mask
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    
    heatmap = np.float32(heatmap) / 255
    cam = 1.0*heatmap + np.float32(img)/255
    cam = cam / np.max(cam)

    img = np.float32(img) / 255
    perturbated = np.multiply(1 - mask, img) + np.multiply(mask, blurred)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
    perturbated = cv2.cvtColor(perturbated, cv2.COLOR_BGR2RGB)

    figure, axes = plt.subplots(nrows=1, ncols=2)
    figure.suptitle(loss)
    for i in range(2):
        axes[i].get_xaxis().set_visible(False)
        axes[i].get_yaxis().set_visible(False)
    axes[0].imshow(img)
    axes[0].set_title('original')
    # axes[1].imshow(perturbated)
    # axes[1].set_title('perturbated')
    # axes[2].imshow(heatmap)
    # axes[2].set_title('heatmap')
    # axes[3].imshow(np.squeeze(mask, axis=2))
    # axes[3].set_title('mask')
    axes[1].imshow(cam)
    axes[1].set_title('cam')
    figure.tight_layout()
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig('pics/'+filename + '_explained.png', bbox_inches='tight', pad_inches=0)
    plt.close()

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

def hook_function(module, input, output):
    np.save('{}_{}_{}'.format(item_name, layer_name, state), output.cpu().data.numpy())
    pass

def process_single_image(model, original_img, verbose=False):
    original_img = cv2.resize(original_img, image_size)
    img = np.float32(original_img) / 255
    blurred_img1 = cv2.GaussianBlur(img, (blur_radius, blur_radius), cv2.BORDER_DEFAULT)
    # blurred_img2 = np.float32(cv2.medianBlur(original_img, blur_radius))/255
    blurred_img_numpy = blurred_img1
    # blurred_img_numpy = blurred_img1.copy()
    mask_init = np.ones((int(image_size[0]*mask_scale),int(image_size[1]*mask_scale)), dtype = np.float32)
    
    # Convert to torch variables
    img = preprocess_image(img)
    blurred_img = preprocess_image(blurred_img_numpy)
    mask = numpy_to_torch(mask_init)

    optimizer = torch.optim.Adam([mask], lr=learning_rate)
    upsample = torch.nn.UpsamplingBilinear2d(size=image_size)
    if use_cuda:
        upsample = upsample.cuda()

    target = torch.nn.Softmax()(model(img))
    category = np.argmax(target.cpu().data.numpy())
    if verbose:
        print("Category with highest probability {}".format(category))
        print("Optimizing.. ")

    prev = 0.0
    for i in range(max_iterations):
        upsampled_mask = upsample(mask)
        # The single channel mask is used with an RGB image, 
        # so the mask is duplicated to have 3 channel,
        upsampled_mask = \
            upsampled_mask.expand(1, 3, upsampled_mask.size(2), \
                                        upsampled_mask.size(3))
        
        # Use the mask to perturbated the input image.
        perturbated_input = img.mul(upsampled_mask) + \
                            blurred_img.mul(1-upsampled_mask)
        
        # noise = np.zeros(image_size+(3,), dtype = np.float32)
        # cv2.randn(noise, 0, 0.2)
        # noise = numpy_to_torch(noise)
        # perturbated_input = perturbated_input + noise
        
        outputs = torch.nn.Softmax()(model(perturbated_input))
        loss = l1_coeff*torch.mean(torch.abs(1 - mask)) + outputs[0, category]
        loss += tv_coeff*tv_norm(mask, tv_beta) if with_tv else 0

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_numpy = loss.cpu().data.numpy()
        print('\tIteration {},\tloss: {:0.6f}'.format(i+1, loss_numpy), end='\r')

        # Optional: clamping seems to give better results
        mask.data.clamp_(0, 1)
    print()

    upsampled_mask = upsample(mask)
    return upsampled_mask, original_img, blurred_img_numpy, loss_numpy, img, blurred_img

if __name__ == '__main__':
    args = parser.parse_args()
    if not args.verbose:
        warnings.filterwarnings("ignore")
    confs = [
        # [blur_radius, with_tv, l1_coeff, tv_coeff, max_iterations, mask_scale],
        [blur_radius, with_tv, l1_coeff, tv_coeff, max_iterations, mask_scale],
        # [blur_radius, with_tv, 0.05, tv_coeff, max_iterations, mask_scale],
        # [blur_radius, with_tv, 0.1, tv_coeff, max_iterations, mask_scale],
        # [blur_radius, with_tv, 0.5, tv_coeff, max_iterations, mask_scale],
        # [blur_radius, with_tv, 1, tv_coeff, max_iterations, mask_scale],
        # [blur_radius, with_tv, l1_coeff, 0.05, max_iterations, mask_scale],
        # [blur_radius, with_tv, l1_coeff, 0.1, max_iterations, mask_scale],
        # [blur_radius, with_tv, l1_coeff, 0.5, max_iterations, mask_scale],
        # [blur_radius, with_tv, l1_coeff, tv_coeff, 100, mask_scale],
        # [blur_radius, with_tv, l1_coeff, tv_coeff, 2000, mask_scale],
        # [blur_radius, with_tv, l1_coeff, tv_coeff, max_iterations, 1],
        # [blur_radius, with_tv, l1_coeff, tv_coeff, max_iterations, 0.75],
        # [blur_radius, with_tv, l1_coeff, tv_coeff, max_iterations, 0.5],
        # [blur_radius, with_tv, l1_coeff, tv_coeff, max_iterations, 0.1],
    ]
    multimasks = (1, 0.75, 0.5, 0.2)

    analyzed_class = range(10)

    model = load_model(args.model)
    # classes = ('plane', 'car', 'bird', 'cat',
    #     'deer', 'dog', 'frog', 'horse', 'ship', 'truck')  
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
    if args.imgpath:
        data = ((cv2.imread(args.imgpath, 1), classes.index(re.sub('[0-9]', '', args.imgpath.split('/')[-1].split('.')[0]))),)
    else:
        to_image = transforms.ToPILImage()
        trainset = torchvision.datasets.CIFAR100(root='./data100', train=False,
                                        download=True, transform=transforms.ToTensor())
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, num_workers=0)
        images, labels = iter(trainloader).next()
        data = []
        for images, labels in iter(trainloader):
            data.extend(list(filter(lambda x: x[1].numpy() in analyzed_class, list(zip(images, labels)))))
    class_counter = collections.Counter()
    class_dict = {}
    for i in data:
        idx = int(i[1].numpy())
        if idx in class_dict.keys():
            class_dict[idx].append(i[0])
        else:
            class_dict[idx] = [i[0]]
    for a_class in range(10):
        items_of_class = class_dict[a_class]
        for i, item in enumerate(items_of_class):
            if i == 2:
                break
            img, label = item, a_class
            if class_counter.get(classes[label], 0) > 1:
                continue
            img = img if args.imgpath else np.array(to_image(img).convert('RGB'))[:, :, ::-1].copy()
            if classes[label] in class_counter:
                class_counter[classes[label]]+=1
            else:
                class_counter.update([classes[label]])
            title = classes[label]+str(class_counter[classes[label]])
            print("Processing {}{},\t{}/{}".format(classes[label], class_counter.get(classes[label], 0), i+1, batch_size))
            for j, conf in enumerate(confs):
                print("\tConf {}/{}".format(j+1, len(confs)))
                if args.multimask:
                    blur_radius, with_tv, l1_coeff, tv_coeff, max_iterations, _ = conf
                    upsampled_masks = []
                    blurred_img_numpys = []
                    losses = []
                    for scale in multimasks:
                        print("Mask scale:\t{}".format(scale))
                        mask_scale = scale
                        upsampled_mask, original_img, blurred_img_numpy, loss, _, _ = process_single_image(model, img, args.verbose)
                        upsampled_masks.append(upsampled_mask)
                        blurred_img_numpys.append(blurred_img_numpy)
                        losses.append(loss)
                    loss = np.mean(losses)
                    _conf = conf
                    _conf[-1] = "multi"
                    save(upsampled_masks, original_img, blurred_img_numpys, title+"".join(["_{}".format(item) for item in _conf]), loss)
                else:
                    blur_radius, with_tv, l1_coeff, tv_coeff, max_iterations, mask_scale = conf
                    upsampled_mask, original_img, blurred_img_numpy, loss, tensor_img, blur_tensor = process_single_image(model, img, args.verbose)
                    save((upsampled_mask,), original_img, (blurred_img_numpy,), title+"".join(["_{}".format(item) for item in conf]), loss)
                    np.save(title, upsampled_mask.cpu().data.numpy())
                    item_name = title
                    for l, lname in ((model.features[32], 'conv5_3'), (model.features[34], 'conv5_4'), (model.classifier[0], 'fc1'), (model.classifier[3], 'fc2'), (model.classifier[6], 'fc3')):
                        layer_name = lname
                        state = 'original'
                        hook = l.register_forward_hook(hook_function)
                        torch.nn.Softmax()(model(tensor_img))
                        state = 'modified'
                        perturbated_input = tensor_img.mul(upsampled_mask) + \
                                blur_tensor.mul(1-upsampled_mask)
                        torch.nn.Softmax()(model(perturbated_input))
                        hook.remove()



