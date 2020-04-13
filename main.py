from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

from utils import print_probs, transform, inverse_transform

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


# CONFIG
# ~~~~~~~~~~~~~~~~~~~

learning_rate = .01
n_iterations = 100
octaves = [7, 8, 9] # For resnet50, octaves range from 0 to 9

# ~~~~~~~~~~~~~~~~~~~

# set mean and std_deviation for imagenet
mean = [.485, .456, .406]
std = [.229, .224, .225]

# initialize net
net = models.resnet50(pretrained=True).to(device)
net.eval()

# preprocess image
img = Image.open("dog1.jpg")
# img.show()
img = transform(img).to(device)
img = torch.unsqueeze(img, 0)

# normalize learning rate for number of octaves
learning_rate = learning_rate / len(octaves)

children = list(net.children())
for i in range(len(octaves)):
    octaves[i] = nn.Sequential(*children)[:octaves[i]-len(children)]

for _ in range(n_iterations):
    for octave in octaves:
        # apply jitter
        y_jitter, x_jitter = np.random.randint(-32, 32, size=2)
        img = torch.roll(img, shifts=(y_jitter, x_jitter), dims=(-2, -1))
        img = img.detach()

        img.requires_grad = True
        logits = octave(img)
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


img = img[0].cpu()
img = inverse_transform(img)

img.show()
