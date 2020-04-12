import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

resnet18 = models.resnet18(pretrained=True)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])

# Note: for first test picture, after ToTensor(), image already loaded in to a range of [0,1]

img = transform(Image.open("waterfall.jpg"))
batch = torch.unsqueeze(img, 0)

resnet18.eval()
probs = F.softmax(resnet18(batch))
print(probs.argmax())