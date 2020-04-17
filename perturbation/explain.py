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

#############################
blur_radius = 11
image_size = (32,32)
tv_beta = 3
learning_rate = 0.1
max_iterations = 500
l1_coeff = 0.01
tv_coeff = 0.2
with_tv = True
mask_scale = 1
#############################

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor
vgg = True

parser = argparse.ArgumentParser()
parser.add_argument('--imgpath', required=False)
parser.add_argument('--model', required=False)
parser.add_argument('--verbose', action='store_true')

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

def save(mask, img, blurred, filename, loss):
    mask = mask.cpu().data.numpy()[0]
    mask = np.transpose(mask, (1, 2, 0))

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

    figure, axes = plt.subplots(nrows=1, ncols=5)
    figure.suptitle(loss)
    for i in range(5):
        axes[i].get_xaxis().set_visible(False)
        axes[i].get_yaxis().set_visible(False)
    axes[0].imshow(img)
    axes[0].set_title('original')
    axes[1].imshow(perturbated)
    axes[1].set_title('perturbated')
    axes[2].imshow(heatmap)
    axes[2].set_title('heatmap')
    axes[3].imshow(np.squeeze(mask, axis=2))
    axes[3].set_title('mask')
    axes[4].imshow(cam)
    axes[4].set_title('cam')
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

def process_single_image(model, original_img, verbose=False):
    original_img = cv2.resize(original_img, image_size)
    img = np.float32(original_img) / 255
    blurred_img1 = cv2.GaussianBlur(img, (blur_radius, blur_radius), cv2.BORDER_DEFAULT)
    # blurred_img2 = np.float32(cv2.medianBlur(original_img, blur_radius))/255
    blurred_img_numpy = blurred_img1
    # blurred_img_numpy = blurred_img1.copy()
    mask_init = np.ones((28,28), dtype = np.float32)
    
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
        
        noise = np.zeros(image_size+(3,), dtype = np.float32)
        cv2.randn(noise, 0, 0.2)
        noise = numpy_to_torch(noise)
        perturbated_input = perturbated_input + noise
        
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
    return upsampled_mask, original_img, blurred_img_numpy, loss_numpy

if __name__ == '__main__':
    args = parser.parse_args()
    if not args.verbose:
        warnings.filterwarnings("ignore")
    confs = [
        # (blur_radius, with_tv, l1_coeff, tv_coeff, max_iterations, mask_scale),
        (blur_radius, with_tv, l1_coeff, tv_coeff, max_iterations, mask_scale),
        (blur_radius, with_tv, 0.05, tv_coeff, max_iterations, mask_scale),
        (blur_radius, with_tv, 0.1, tv_coeff, max_iterations, mask_scale),
        (blur_radius, with_tv, 0.5, tv_coeff, max_iterations, mask_scale),
        (blur_radius, with_tv, 1, tv_coeff, max_iterations, mask_scale),
        (blur_radius, with_tv, l1_coeff, 0.05, max_iterations, mask_scale),
        (blur_radius, with_tv, l1_coeff, 0.1, max_iterations, mask_scale),
        (blur_radius, with_tv, l1_coeff, 0.5, max_iterations, mask_scale),
        (blur_radius, with_tv, l1_coeff, tv_coeff, 100, mask_scale),
        (blur_radius, with_tv, l1_coeff, tv_coeff, 2000, mask_scale),
        # (blur_radius, with_tv, l1_coeff, tv_coeff, max_iterations, 1),
    ]

    model = load_model(args.model)
    if args.imgpath:
        img = cv2.imread(args.imgpath, 1)
        for conf in confs:
            blur_radius, with_tv, l1_coeff, tv_coeff, max_iterations, mask_scale = conf
            upsampled_mask, original_img, blurred_img_numpy, loss = process_single_image(model, img)
            save(upsampled_mask, original_img, blurred_img_numpy, args.imgpath.split('/')[-1].split('.')[0]+"".join(["_{}".format(item) for item in conf]), loss)
    else:
        to_image = transforms.ToPILImage()
        trainset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transforms.ToTensor())
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=50,
                                                shuffle=True, num_workers=0)
        images, labels = iter(trainloader).next()
        data = list(zip(images, labels))
        classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')  
        class_counter = collections.Counter()
        for i, item in enumerate(data):
            img, label = item
            if class_counter.get(classes[label], 0) > 3:
                continue
            img = np.array(to_image(img).convert('RGB'))[:, :, ::-1].copy()
            if classes[label] in class_counter:
                class_counter[classes[label]]+=1
            else:
                class_counter.update([classes[label]])
            title = classes[label]+str(class_counter[classes[label]])
            print("Processing {}{},\t{}/50".format(classes[label], class_counter.get(classes[label], 0), i+1))
            for j, conf in enumerate(confs):
                print("\tConf {}/{}".format(j+1, len(confs)))
                blur_radius, with_tv, l1_coeff, tv_coeff, max_iterations, mask_scale = conf
                upsampled_mask, original_img, blurred_img_numpy, loss = process_single_image(model, img, args.verbose)
                save(upsampled_mask, original_img, blurred_img_numpy, title+"".join(["_{}".format(item) for item in conf]), loss)

