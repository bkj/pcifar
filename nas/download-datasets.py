#!/usr/bin/env python

"""
    download-datasets.py
    
    Download (most of) the `torchvision` datasets 
"""


import os
import sys
from torchvision import datasets

root = './data'

name = 'MNIST'
print >> sys.stderr, name
_ = datasets.MNIST(os.path.join(root, name), download=True)

name = 'fasionMNIST'
print >> sys.stderr, name
_ = datasets.FashionMNIST(os.path.join(root, name), download=True)

name = 'STL10'
print >> sys.stderr, name
_ = datasets.STL10(os.path.join(root, name), download=True)

name = 'CIFAR10'
print >> sys.stderr, name
_ = datasets.CIFAR10(os.path.join(root, name), download=True)
# _ = datasets.CIFAR100(os.path.join(root, name), download=True)

name = 'SVHN'
print >> sys.stderr, name
_ = datasets.SVHN(os.path.join(root, name), download=True)

# !! I think this is too big for current purposes
# name = 'LSUN'
# print >> sys.stderr, name
# _ = datasets.LSUN(os.path.join(root, name), download=True)
# _ = datasets.CIFAR100(os.path.join(root, name), download=True)
