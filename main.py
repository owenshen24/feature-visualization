from typing import Dict, List, Optional
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import scipy.ndimage as nd
import psutil

from utils import print_probs, transform, inverse_transform

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
# initialize net and set to evaluation mode
net = models.googlenet(pretrained=True).to(device);
net.eval()

#~~~~~~~ CONFIG ~~~~~~~~~~

learning_rate = 0.25
n_iterations = 100
n_octaves = 3
octave_scale = 1.4

# Each layer in this list represents the first however many layers of the net
layers = [10,11,12] 

#~~~~~~~~~~~~~~~~~~~~~~~~~

# set image color mean and std_deviation for imagenet
mean = [.485, .456, .406]
std = [.229, .224, .225]

children = list(net.children())
for i in range(len(layers)):
    layers[i] = nn.Sequential(*children)[:layers[i]]

# preprocess image
img = Image.open("images/dog5.jpg")
# img.show()
img = transform(img).to(device)
img = torch.unsqueeze(img, 0)

#img = torch.rand(1, 3, 500, 500)

# deep copy the image
img_copy = img.clone().detach()

# normalize learning rate
# learning_rate = learning_rate / (len(layers)*n_octaves)


def optimize_image(img, n_iterations, layers, learning_rate):
    for _ in range(n_iterations):
        for layer in layers:
            # apply jitter
            y_jitter, x_jitter = np.random.randint(-32, 32, size=2)
            img = torch.roll(img, shifts=(y_jitter, x_jitter), dims=(-2, -1))
            img = img.detach()

            img.requires_grad = True
            logits = layer(img)
            loss = -(logits**2).mean()
            #loss = -(logits[0][0]**2).mean()
            loss.backward()

            g = img.grad.data
            g = g/g.abs().mean()
            img = img - learning_rate*g

            # Normalize image 
            # from https://github.com/eriklindernoren/PyTorch-Deep-Dream
            for c in range(3):
                m, s = mean[c], std[c]
                img[0][c] = torch.clamp(img[0][c], -m/s, (1-m)/s)
            
            # undo jitter
            img = torch.roll(img, shifts=(-y_jitter, -x_jitter), dims=(-2, -1))
    
    return img

def deep_dream_with_octaves(img, n_iterations, layers,
                            learning_rate, n_octaves, octave_scale):
    
    # Each item of octave_imgs
    # is a zoomed-in (i.e. lower-res) version of the previous image
    octave_imgs = [img[0].cpu().numpy()]
    for i in range(n_octaves-1):
        new_octave_img = nd.zoom(octave_imgs[-1], 
                                 (1, 1.0/octave_scale, 1.0/octave_scale),
                                 order=2)
        octave_imgs.append(new_octave_img)
    for i in range(len(octave_imgs)):
        octave_imgs[i] = torch.tensor(octave_imgs[i]).unsqueeze(0).float().to(device)

    for octave, octave_img in enumerate(octave_imgs):
        im = img[0].cpu()
        im = inverse_transform(im)
        h, w = octave_img.shape[-2:]
        if octave > 0:
            # upscale previous octave's details
            h1, w1 = detail.shape[-2:]
            detail = detail[0].cpu().detach().numpy()
            detail = nd.zoom(detail, (1, 1.0*h/h1, 1.0*w/w1), order=1)
            detail = torch.tensor(detail).unsqueeze(0).float().to(device)
            img = octave_img + detail
        else:
            img = octave_img

        img = optimize_image(n_iterations, img, layers, learning_rate)
        detail = img-octave_img
    
    return img

    # Make the list low to high res
    octave_imgs.reverse()




print(print_probs(F.softmax(net(img), dim=1)[0]))
# print_probs(F.softmax(net(img)))

for proc in psutil.process_iter():
    if proc.name() == "display":
        proc.kill()

img = img[0].cpu()
img = inverse_transform(img)
img.show()

time.sleep(5)