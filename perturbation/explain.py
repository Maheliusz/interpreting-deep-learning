import torch
from torch.autograd import Variable
from torchvision import models
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse

#############################
blur_radius = 11
ksd = 5
image_size = (32,32)
tv_beta = 3
learning_rate = 0.1
max_iterations = 500
l1_coeff = 0.01
tv_coeff = 0.2
#############################

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor

parser = argparse.ArgumentParser()
parser.add_argument('imgpath')
parser.add_argument('--model', required=False)

def tv_norm(input, tv_beta):
    img = input[0, 0, :]
    row_grad = torch.mean(torch.abs((img[:-1 , :] - img[1 :, :])).pow(tv_beta))
    col_grad = torch.mean(torch.abs((img[: , :-1] - img[: , 1 :])).pow(tv_beta))
    return row_grad + col_grad

def preprocess_image(img):
    means=[0.485, 0.456, 0.406]
    stds=[0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[: , :, ::-1]
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

def save(mask, img, blurred, filename):
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
    figure.suptitle(filename.split('.')[0])
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
    plt.savefig(filename.split('.')[0] + '_explained.png')

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
    model.eval()
    if use_cuda:
        model.cuda()
    
    if not model_path:
        for p in model.features.parameters():
            p.requires_grad = False
        for p in model.classifier.parameters():
            p.requires_grad = False

    return model

if __name__ == '__main__':
    args = parser.parse_args()

    model = load_model(args.model)
    imgpath = args.imgpath
    
    original_img = cv2.imread(imgpath, 1)
    original_img = cv2.resize(original_img, image_size)
    img = np.float32(original_img) / 255
    blurred_img1 = cv2.GaussianBlur(img, (blur_radius, blur_radius), ksd)
    blurred_img2 = np.float32(cv2.medianBlur(original_img, blur_radius))/255
    blurred_img_numpy = (blurred_img1 + blurred_img2) / 2
    # blurred_img_numpy = blurred_img1.copy()
    mask_init = np.ones((28,28), dtype = np.float32)
    
    # Convert to torch variables
    img = preprocess_image(img)
    blurred_img = preprocess_image(blurred_img_numpy)
    mask = numpy_to_torch(mask_init)

    if use_cuda:
        upsample = torch.nn.UpsamplingBilinear2d(size=image_size).cuda()
    else:
        upsample = torch.nn.UpsamplingBilinear2d(size=image_size)
    optimizer = torch.optim.Adam([mask], lr=learning_rate)

    target = torch.nn.Softmax()(model(img))
    category = np.argmax(target.cpu().data.numpy())
    print("Category with highest probability {}".format(category))
    print("Optimizing.. ")

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
        loss = l1_coeff*torch.mean(torch.abs(1 - mask)) + \
                tv_coeff*tv_norm(mask, tv_beta) + outputs[0, category]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Optional: clamping seems to give better results
        mask.data.clamp_(0, 1)

    upsampled_mask = upsample(mask)
    save(upsampled_mask, original_img, blurred_img_numpy, sys.argv[1])
