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
# analyzed_class = 0
# analyzed_plane = 16

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

def to_cam(img, mask):
    # mask = np.transpose(mask, (1, 2, 0))
    # print("{};{}".format(img.shape, mask.shape))
    mask = (mask - np.min(mask)) / np.max(mask)
    mask = 1 - mask
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    img = img + np.min(img)
    img = np.float32(img) / 255
    # img = np.abs(img)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # print(img)
    cam = 1.0*heatmap + np.float32(img)/255
    cam = cam / np.max(cam)
    return cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)

def save(masks, img, blurs, filename, loss):
    mask = None
    for item in masks:
        _mask = (item.cpu() if not use_cuda else item.cuda()).data.numpy()[0]
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

def plot_to_file(data, filename):
    fig = plt.figure(figsize=(data.shape[1]/plt.gcf().dpi, data.shape[0]/plt.gcf().dpi))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(data, interpolation='none')
    ax.set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0)
    plt.margins(0,0)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    if filename is not None:
        plt.savefig(filename+'.png')#, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.show()

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
    # print(output.shape)
    global vis
    vis = output.reshape(output.shape[-3:])
    # vis = output.reshape(-1, output.shape[-1])
    np_vis = (vis.cpu() if not use_cuda else vis.cuda()).data.numpy()[:]
    # blurred_activations = []
    # for i in range(np_vis.shape[0]):
    #     blurred_activations.append(cv2.GaussianBlur(np_vis[i], (blur_radius, blur_radius), cv2.BORDER_DEFAULT))
    # blurred_activations = np.expand_dims([cv2.GaussianBlur(_vis, (blur_radius, blur_radius), cv2.BORDER_DEFAULT) for _vis in np_vis], 0)
    blurred_activations = np.zeros(output.shape, dtype=np.float32)
    # blurred_activations = np.expand_dims(blurred_activations, 0)
    # print(vis.shape)
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

    # print(model)
    # exit()
    
    if not model_path:
        for p in model.features.parameters():
            p.requires_grad = False
        for p in model.classifier.parameters():
            p.requires_grad = False

    # model.conv1.register_forward_hook(hook_function)
    # model.classifier[6].register_forward_hook(hook_function)
    # model.features[25].register_forward_hook(infohook)
    # global vis
    # shape = np.subtract(image_size, model.conv2.kernel_size) + 1
    # vis = np.ones((1, 1000))
    # vis = np.ones((1, 16, 10, 10))

    return model

def process_single_image(model, original_img, verbose=False):
    original_img = cv2.resize(original_img, image_size)
    img = np.float32(original_img) / 255
    blurred_img1 = cv2.GaussianBlur(img, (blur_radius, blur_radius), cv2.BORDER_DEFAULT)
    # blurred_img2 = np.float32(cv2.medianBlur(original_img, blur_radius))/255
    blurred_img_numpy = blurred_img1
    # blurred_img_numpy = blurred_img1.copy()
    mask_init = np.ones((int(image_size[0]*mask_scale),int(image_size[1]*mask_scale)), dtype = np.float32)

    # inner layers mask initializations
    global l1_mask
    l1_mask = np.ones(vis.shape[-3:], dtype=np.float32)
    # print(vis.shape)
    
    # Convert to torch variables
    img = preprocess_image(img)
    blurred_img = preprocess_image(blurred_img_numpy)
    mask = numpy_to_torch(mask_init)
    # mask = numpy_to_torch(l1_mask)
    l1_mask = torch.from_numpy(l1_mask)
    # if use_cuda:
    #     l1_mask = l1_mask.cuda()
    l1_mask = Variable(l1_mask, requires_grad = True)

    optimizer = torch.optim.Adam([l1_mask], lr=learning_rate)
    # upsample = torch.nn.UpsamplingBilinear2d(size=image_size)
    upsample = torch.nn.UpsamplingBilinear2d(size=(224,224))
    if use_cuda:
        upsample = upsample.cuda()

    # img = upsample(img)
    target = torch.nn.Softmax()(model(img))
    category = np.argmax((target.cpu() if not use_cuda else target.cuda()).data.numpy())
    if verbose:
        print("Category with highest probability {}".format(category))
        print("Optimizing.. ")

    prev = 0.0
    for i in range(max_iterations):
        # upsampled_mask = upsample(mask)
        # The single channel mask is used with an RGB image, 
        # so the mask is duplicated to have 3 channel,
        # upsampled_mask = \
        #     upsampled_mask.expand(1, 3, upsampled_mask.size(2), \
        #                                 upsampled_mask.size(3))
        # l1_mask = \
        #     l1_mask.expand(1, 6, l1_mask.size(2), \
        #                                 l1_mask.size(3))
        
        # Use the mask to perturbated the input image.
        # perturbated_input = img.mul(upsampled_mask) + \
        #                     blurred_img.mul(1-upsampled_mask)
        
        # noise = np.zeros(image_size+(3,), dtype = np.float32)
        # cv2.randn(noise, 0, 0.2)
        # noise = numpy_to_torch(noise)
        # perturbated_input = perturbated_input + noise
        
        # outputs = torch.nn.Softmax()(model(perturbated_input))
        outputs = torch.nn.Softmax()(model(img))
        # loss = outputs[0, category]
        loss = l1_coeff*torch.mean(torch.abs(1 - l1_mask)) + outputs[0, category]
        for j in range(l1_mask.shape[0]):
            loss += tv_coeff*tv_norm(l1_mask[j], tv_beta) if with_tv else 0
        # loss += tv_coeff*tv_norm(l1_mask, tv_beta) if with_tv else 0

        optimizer.zero_grad()
        # print(loss, end='\r')
        loss.backward()
        optimizer.step()
        loss_numpy = (loss.cpu() if not use_cuda else loss.cuda()).data.numpy()
        print('\tIteration {},\tloss: {:0.6f}'.format(i+1, loss_numpy), end='\r')

        # Optional: clamping seems to give better results
        # mask.data.clamp_(0, 1)
        l1_mask.data.clamp_(0, 1)
    np_vis = (vis.cpu() if not use_cuda else vis.cuda()).data.numpy()
    # np_vis = np_vis.reshape(-1, np_vis.shape[-2], np_vis.shape[-1])
    masks = (l1_mask.cpu() if not use_cuda else l1_mask.cuda()).data.numpy()#.reshape(-1, np_vis.shape[-2], np_vis.shape[-1])
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
    return None, original_img, blurred_img_numpy, loss_numpy
    # return l1_mask, vis, vis, loss_numpy


def run(model, confs, analyzed_class, analyzed_plane, imgpath=None):
    global l1_mask, vis
    l1_mask = np.ones(mask_size)
    vis = np.ones(mask_size)
    # hook = model.features[analyzed_plane].register_forward_hook(hook_function)
    hook = model.classifier[0].register_forward_hook(hook_function)
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
        # images, labels = iter(trainloader).next()
        # print(labels[0])
        # exit()
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
    # print(class_dict.keys())
    # exit()
    #TODO
    # for analyzed_class in range(10):
    items_of_class = class_dict[analyzed_class]
    # global class_masks
    # class_masks = []
    for i, item in enumerate(items_of_class):
        if i == 25:
            break
        img, label = item, analyzed_class
        # if class_counter.get(classes[label], 0) > 1:
        #     continue
        img = img if args.imgpath else np.array(to_image(img).convert('RGB'))[:, :, ::-1].copy()

        # print(classes[label])
        # plt.imshow(img)
        # plt.show()
        # break

        if classes[label] in class_counter:
            class_counter[classes[label]]+=1
        else:
            class_counter.update([classes[label]])
        title = classes[label]+str(class_counter[classes[label]])
        print("Processing {}{},\t{}/{}".format(classes[label], class_counter.get(classes[label], 0), i+1, len(items_of_class)))
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
                    upsampled_mask, original_img, blurred_img_numpy, loss = process_single_image(model, img, args.verbose)
                    upsampled_masks.append(upsampled_mask)
                    blurred_img_numpys.append(blurred_img_numpy)
                    losses.append(loss)
                loss = np.mean(losses)
                _conf = conf
                _conf[-1] = "multi"
                # save(upsampled_masks, original_img, blurred_img_numpys, title+"".join(["_{}".format(item) for item in _conf]), loss)
            else:
                blur_radius, with_tv, l1_coeff, tv_coeff, max_iterations, mask_scale = conf
                upsampled_mask, original_img, blurred_img_numpy, loss = process_single_image(model, img, args.verbose)
                # save((upsampled_mask,), original_img, (blurred_img_numpy,), title+"".join(["_{}".format(item) for item in conf]), loss)
    # global class_masks
    np.save('pics/cifar100/class_{}_{}'.format(classes[analyzed_class], analyzed_plane), class_masks)
    # _class_masks = np.array([np.concatenate(pic, axis=1) for pic in class_masks])
    # _class_masks_reshaped = np.concatenate(_class_masks, axis=0)
    # print(_class_masks_reshaped.shape)
    # plot_to_file(_class_masks_reshaped, 'pics/class_{}_{}'.format(classes[analyzed_class], analyzed_plane))
    # _class_masks_aggregated = np.sum(_class_masks, axis=0)
    # plot_to_file(_class_masks_aggregated, 'pics/class_{}_agg_{}'.format(classes[analyzed_class], analyzed_plane))
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
    # print(model)
    # exit()
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
    coarse_classes = ["aquatic_mammals"	, "fish", "flowers", "food_containers", "fruit_and_vegetables", "household_electrical_devices", "household_furniture"	, "insects"	, "large_carnivores", "large_man-made_outdoor_things", "large_natural_outdoor_scenes", "large_omnivores_and_herbivores", "medium-sized_mammals", "non-insect_invertebrates", "people", "reptiles", "small_mammals", "trees", "vehicles_1", "vehicles_2",] 
    fine_class_to_coarse = {
        "beaver":0, "dolphin":0, "otter":0, "seal":0, "whale":0,
        "aquarium_fish":1, "flatfish":1, "ray":1, "shark":1, "trout":1,
        "orchids":2, "poppies":2, "roses":2, "sunflowers":2, "tulips":2,
        "bottles":3, "bowls":3, "cans":3, "cups":3, "plates":3,
        "apples":4, "mushrooms":4, "oranges":4, "pears":4, "sweet_peppers":4,
        "clock":5, "computer_keyboard":5, "lamp":5, "telephone":5, "television":5,
        "bed":6, "chair":6, "couch":6, "table":6, "wardrobe":6,
        "bee":7, "beetle":7, "butterfly":7, "caterpillar":7, "cockroach":7,
        "bear":8, "leopard":8, "lion":8, "tiger":8, "wolf":8,
        "bridge":9, "castle":9, "house":9, "road":9, "skyscraper":9,
        "cloud":10, "forest":10, "mountain":10, "plain":10, "sea":10,
        "camel":11, "cattle":11, "chimpanzee":11, "elephant":11, "kangaroo":11,
        "fox":12, "porcupine":12, "possum":12, "raccoon":12, "skunk":12,
        "crab":13, "lobster":13, "snail":13, "spider":13, "worm":13,
        "baby":14, "boy":14, "girl":14, "man":14, "woman":14,
        "crocodile":15, "dinosaur":15, "lizard":15, "snake":15, "turtle":15,
        "hamster":16, "mouse":16, "rabbit":16, "shrew":16, "squirrel":16,
        "maple":17, "oak":17, "palm":17, "pine":17, "willow":17,
        "bicycle":18, "bus":18, "motorcycle":18, "pickup_truck":18, "train":18,
        "lawn_mower":19, "rocket":19, "streetcar":19, "tank":19, "tractor":19,
    }
    # if analyzed_plane in (34, 32, 30, 28):
    #     mask_size= (1, 512, 2, 2)
    # elif analyzed_plane == 25:
    #     mask_size = (1, 512, 4, 4)
    # elif analyzed_plane == 16:
    #     mask_size = (1, 256, 8, 8)
    mask_size = (1, 4096)
    # nums = []
    # for k, v in fine_class_to_coarse.items():
    #     if v==0:
    #         print(k)
    #         nums.append(classes.index(k))
    # for i, item in enumerate(classes):
    #     print('{}:\t{}'.format(i, item))
    # exit()
    for i in range(10):
        for j in (0,):# 25, 16):
            class_masks = []
            run(model, confs, i, j, args.imgpath)
