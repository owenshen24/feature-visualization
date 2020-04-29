from typing import Optional, List
from torchvision import transforms

def print_probs(probs, print_zeros: Optional[bool] = False):
    '''Prints distribution of Imagenet classes under given distribution
    print_zeros: whether or not to print low probability classes
    '''

    
    assert len(probs) == 1000, "probabilities must be length 1000"
    assert abs(sum(probs)-1) < .01, "probabilities must sum to near 1"

    with open('imagenet_classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]

    probs = [float(p) for p in probs]
    
    d = dict()
    for i in range(1000):
        d[probs[i]] = classes[i]
    
    sorted_probs = sorted(probs, reverse=True)

    for p in sorted_probs:
        if print_zeros or round(float(100*p), 2) != 0:
            print(f'{round(float(100*p), 2):<5} | {d[p]}')

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