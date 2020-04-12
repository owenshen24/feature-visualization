from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

from utils import print_probs


resnet50 = models.resnet50(pretrained=True)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])

# Note: for first test picture, after ToTensor(), 
# image already loaded in to a range of [0,1]

img = transform(Image.open("dog3.jpg"))
batch = torch.unsqueeze(img, 0)

resnet50.eval()
probs = F.softmax(resnet50(batch), dim=1)

print_probs(probs[0])

