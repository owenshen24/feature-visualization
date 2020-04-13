from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import numpy as np

from utils import print_probs

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

resnet50 = models.resnet50(pretrained=True).to(device)


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

transform = transforms.Compose([
    # transforms.Resize(256),
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

# Note: for first test picture, after ToTensor(),
# image already loaded in to a range of [0,1]

img = transform(Image.open("dog3.jpg")).to(device)
img = torch.unsqueeze(img, 0)

img.requires_grad = True
optimizer = optim.SGD([img], lr=.01)
resnet50.eval()

for _ in range(1):
    height, width = np.random.randint(-32, 32, size=2)
    torch.roll(img, shifts=(height, width), dims=(-2, -1))

    logits = resnet50(img)
    loss = -(logits**2).sum()
    optimizer.zero_grad()
    loss.backward()
    g = img.grad
    img.grad = g/(g.abs().mean())
    optimizer.step()

img = img[0].cpu()
img = inverse_transform(img)
img.show()
