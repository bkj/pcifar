#!/usr/bin/env python

"""
    download-datasets.py
    
    - Download (most of) the `torchvision` datasets
    - Convert to directories of images
        - That way, can use `ImageFolder` dataloader for all datasets
"""


import os
import sys
from tqdm import tqdm
from torchvision import datasets

# --
# Helpers

def save(name, dataset, mode):
    for i, (img, lab) in tqdm(enumerate(dataset), total=len(dataset)):
        outdir = os.path.join(root, 'manual', name, mode, str(lab))
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        
        img.save(os.path.join(outdir, '%d.jpg' % i))


root = './data'

name = 'MNIST'
print >> sys.stderr, name
save(name, datasets.MNIST(os.path.join(root, name), download=True, train=True), 'train')
save(name, datasets.MNIST(os.path.join(root, name), download=True, train=False), 'test')


name = 'fashionMNIST'
print >> sys.stderr, name
save(name, datasets.FashionMNIST(os.path.join(root, name), download=True, train=True), 'train')
save(name, datasets.FashionMNIST(os.path.join(root, name), download=True, train=False), 'test')


name = 'STL10'
print >> sys.stderr, name
save(name, datasets.STL10(os.path.join(root, name), download=True, split='train'), 'train')
save(name, datasets.STL10(os.path.join(root, name), download=True, split='test'), 'test')


name = 'CIFAR10'
print >> sys.stderr, name
save(name, datasets.CIFAR10(os.path.join(root, name), download=True, train=True), 'train')
save(name, datasets.CIFAR10(os.path.join(root, name), download=True, train=False), 'test')


name = 'CIFAR100'
print >> sys.stderr, name
save(name, datasets.CIFAR100(os.path.join(root, name), download=True, train=True), 'train')
save(name, datasets.CIFAR100(os.path.join(root, name), download=True, train=False), 'test')


name = 'SVHN'
print >> sys.stderr, name
save(name, datasets.SVHN(os.path.join(root, name), download=True, split='train'), 'train')
save(name, datasets.SVHN(os.path.join(root, name), download=True, split='test'), 'test')
