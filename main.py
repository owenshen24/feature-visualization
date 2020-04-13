from typing import Dict, List, Optional
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import scipy.ndimage as nd
import matplotlib.pyplot as plt
import psutil

from utils import print_probs, transform, inverse_transform

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


# CONFIG
# ~~~~~~~~~~~~~~~~~~~

learning_rate = .04
n_iterations = 100
layers = [6,7,8,9] # For resnet50, layer can range from 0 to 9
n_octaves = 4
octave_scale = 1.4

# ~~~~~~~~~~~~~~~~~~~

# set mean and std_deviation for imagenet
mean = [.485, .456, .406]
std = [.229, .224, .225]

# initialize net
net = models.resnet50(pretrained=True).to(device)
net.eval()

# # preprocess image
# img = Image.open("dog1.jpg")
# # img.show()
# img = transform(img).to(device)
# img = torch.unsqueeze(img, 0)

img = torch.rand(1, 3, 255, 255)

# normalize learning rate
learning_rate = learning_rate / (len(layers)*n_octaves)

children = list(net.children())
for i in range(len(layers)):
    layers[i] = nn.Sequential(*children)[:layers[i]-len(children)]

octave_imgs = [img[0].cpu().numpy()]
for i in range(n_octaves-1):
    new_octave_img = nd.zoom(octave_imgs[-1], (1, 1.0/octave_scale, 1.0/octave_scale),
                         order=1)
    octave_imgs.append(new_octave_img)
for i in range(len(octave_imgs)):
    octave_imgs[i] = torch.tensor(octave_imgs[i]).unsqueeze(0).float().to(device)
octave_imgs.reverse()


for octave, octave_img in enumerate(octave_imgs):
    im = img[0].cpu()
    im = inverse_transform(im)
    im.show()
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

    for _ in range(n_iterations):
        for layer in layers:
            # apply jitter
            y_jitter, x_jitter = np.random.randint(-32, 32, size=2)
            img = torch.roll(img, shifts=(y_jitter, x_jitter), dims=(-2, -1))
            img = img.detach()

            img.requires_grad = True
            logits = layer(img)
            loss = -(logits**2).mean()
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

    detail = img-octave_img

# print(print_probs(F.softmax(net(img), dim=1)[0]))
# print_probs(F.softmax(net(img)))

img = img[0].cpu()
img = inverse_transform(img)
img.show()

time.sleep(5)

for proc in psutil.process_iter():
    if proc.name() == "display":
        proc.kill()
