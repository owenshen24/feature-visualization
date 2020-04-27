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

learning_rate = 10
n_iterations = 100
layers = [11] # For resnet50, layer can range from 0 to 9
# n_octaves = 7
# octave_scale = 1.4
# channel = 0

# ~~~~~~~~~~~~~~~~~~~

# set mean and std_deviation for imagenet
mean = [.485, .456, .406]
std = [.229, .224, .225]

# initialize net
net = models.googlenet(pretrained=True).to(device)
net.eval()

# preprocess image
img = Image.open("dog1.jpg")
img.show()
img = transform(img).unsqueeze(0).to(device)


children = list(net.children())
for i in range(len(layers)):
    layers[i] = nn.Sequential(*children)[:layers[i]-len(children)]


for _ in range(n_iterations):
    for layer in layers:
        # apply jitter
        y_jitter, x_jitter = np.random.randint(-5, 5, size=2)


        img = torch.roll(img, shifts=(y_jitter, x_jitter), dims=(-2, -1))
        img = img.detach()

        params = torch.rfft(img, signal_ndim=2).to(device)
        params.requires_grad = True
        img = torch.irfft(params, signal_ndim=2)

        # img = torch.irfft(params, signal_ndim=2).unsqueeze(0).to(device)
        out = layer(img)
        loss = -(out[0]**2).mean()
        loss.backward()


        g = params.grad.data
        g = g/g.abs().mean()
        params = params - learning_rate*g

        img = torch.irfft(params, signal_ndim=2)

        # Normalize image 
        # from https://github.com/eriklindernoren/PyTorch-Deep-Dream
        for c in range(3):
            m, s = mean[c], std[c]
            img[0][c] = torch.clamp(img[0][c], -m/s, (1-m)/s)
        
        # undo jitter
        img = torch.roll(img, shifts=(-y_jitter, -x_jitter), dims=(-2, -1))

# print(print_probs(F.softmax(net(img), dim=1)[0]))


# for proc in psutil.process_iter():
#     if proc.name() == "display":
#         proc.kill()

img = img[0].cpu()
img = inverse_transform(img)
img.show()

time.sleep(5)

