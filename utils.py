from typing import Optional, List

import torch
import torch.nn as nn
from torchvision import transforms



# with open('imagenet_classes.txt') as f:
#     classes = [line.strip() for line in f.readlines()]

# def print_probs(probs, print_zeros: Optional[bool] = False):
#     '''Prints distribution of Imagenet classes under given distribution
#     print_zeros: whether or not to print low probability classes
#     '''

#     global classes

#     assert len(probs) == 1000, "probabilities must be length 1000"
#     assert abs(sum(probs)-1) < .01, "probabilities must sum to near 1"

#     probs = [float(p) for p in probs]
    
#     d = dict()
#     for i in range(1000):
#         d[probs[i]] = classes[i]
    
#     sorted_probs = sorted(probs, reverse=True)

#     for p in sorted_probs:
#         if print_zeros or round(float(100*p), 2) != 0:
#             print(f'{round(float(100*p), 2):<5} | {d[p]}')

class Hook:

    def __init__(self, module: nn.Module, backward: Optional[bool] = False, hook_fn: Optional = None):

        hook_fn =  self.hook_fn if hook_fn is None else hook_fn

        if backward is False:
            self.hook = module.register_forward_hook(hook_fn)
        else:
            self.hook = module.register_hook(hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output
        assert False # this is to pop out of network evaluation;
                     # you don't need to compute later activations

    def close(self):
        self.hook.remove()


transform = transforms.Compose([
    # transforms.Resize(512),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])

# https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/3
inverse_transform = transforms.Compose([
    transforms.Normalize(mean = [ 0., 0., 0. ],
                         std = [ 1/0.229, 1/0.224, 1/0.225 ]),
    transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                         std = [ 1., 1., 1. ]),
    transforms.ToPILImage()
    ])