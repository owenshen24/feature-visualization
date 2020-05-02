from typing import Dict, List, Tuple, Optional
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

from utils import transform, inverse_transform, Hook

from guided_filter_pytorch.guided_filter import GuidedFilter

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def normalize_img(img: torch.Tensor) -> None:
    # mean and std_deviation for imagenet
    mean = [.485, .456, .406]
    std = [.229, .224, .225]

    # from https://github.com/eriklindernoren/PyTorch-Deep-Dream
    for c in range(3):
        m, s = mean[c], std[c]
        img[0][c].clamp(-m/s, (1-m)/s)


def optimize(img: torch.Tensor, net: nn.Module, objective: str, layer: str,
      channel: Optional[int] = None, neuron_coord: Optional[Tuple[int]]= None, ) -> torch.Tensor:
    global device
    
    assert len(img.shape) == 4 and img.shape[1] == 3, "input must be image(s)"
    assert objective in {"deepdream", "channel", "neuron"}, \
            "objective must be 'deepdream', 'channel', or 'neuron'"
    # assert layer in dict(net.named_parameters()).keys()
    
    if objective is "channel":
        assert isinstance(channel, int)
    elif objective is "neuron":
        assert isinstance(channel, int)
        assert isinstance(neuron_coord, Tuple[int])
        assert len(neuron_coord) == 2
        assert neuron[0] < img.shape[1] and neuron[1] < img.shape[2]

    def loss_fn(output: torch.Tensor, objective: str) -> torch.Tensor:
        if objective is "deepdream":
            loss = ((output)**2).mean()
        elif objective is "channel":
            loss = output[:,channel].mean()
        elif objective is "neuron":
            loss = output[:,channel,neuron_coord[0], neuron_coord[1]].mean()

        return loss

    def total_variation(img: torch.Tensor, beta=1.4) -> torch.Tensor:
        differences = ((torch.roll(img, shifts=(0,1), dims=(-2,-1))-img)**2+
                        (torch.roll(img, shifts=(1,0), dims=(-2,-1))-img)**2)**(beta/2)
        out = differences.mean()
        return out

    gf = GuidedFilter(3, .2)
    guided_filter = lambda x: gf(x,x)

    module = net
    for l in layer.split("."):
        module = module._modules[l]

    activation_hook = Hook(module)

    normalize_img(img)
    img.requires_grad = True
    grad_hook = Hook(img, backward=True, 
                     hook_fn = lambda x: x/(x.abs().mean()+1e-4))

    optimizer = torch.optim.Adam([img], lr=.05)

    for i in range(100):
        # apply jitter
        y_jitter, x_jitter = np.random.randint(-32, 32, size=2)
        img.roll(shifts=(y_jitter, x_jitter), dims=(-2, -1))

        # the activation hook throws an error, so the net only processes
        # as much as it needs to in order to get the activation_hook.output
        try:
            _ = net(guided_filter(img))
        except:
            pass

        # print(loss_fn(activation_hook.output, objective))
        # print(total_variation(img))
        loss = loss_fn(activation_hook.output, objective) + .01*total_variation(img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        normalize_img(img)
        
        # undo jitter
        img.roll(shifts=(-y_jitter, -x_jitter), dims=(-2, -1))
    grad_hook.close()
    activation_hook.close()

    return img



net = models.googlenet(pretrained=True).to(device)
net.eval()


# preprocess image
# img = Image.open("dog1.jpg")
# img = transform(img).to(device)
# img = torch.unsqueeze(img, 0)

img = torch.rand(1, 3, 144, 144).to(device)

img = optimize(img=img, net=net, objective="channel",
               layer='inception3a.branch1', channel=0)

img = img[0].cpu()
img = inverse_transform(img)
img.show()

time.sleep(20)

for proc in psutil.process_iter():
    if proc.name() == "display":
        proc.kill()

