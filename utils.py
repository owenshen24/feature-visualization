from typing import Optional, List


with open('imagenet_classes.txt') as f:
    classes = [line.strip() for line in f.readlines()]

def print_probs(probs, print_zeros: Optional[bool] = False):
    '''Prints distribution of Imagenet classes under given distribution
    print_zeros: whether or not to print low probability classes
    '''

    global classes

    assert len(probs) == 1000, "probabilities must be length 1000"
    assert abs(sum(probs)-1) < .01, "probabilities must sum to near 1"

    probs = [float(p) for p in probs]
    
    d = dict()
    for i in range(1000):
        d[probs[i]] = classes[i]
    
    sorted_probs = sorted(probs, reverse=True)

    for p in sorted_probs:
        if print_zeros or round(float(100*p), 2) != 0:
            print(f'{round(float(100*p), 2):<5} | {d[p]}')