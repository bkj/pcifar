#!/usr/bin/env python

"""
    data.py
    
    Data loaders for NAS
"""

import torch
import torchvision
from torchvision import transforms, datasets

def basic(name, center=False, hflip=False):
    
    transform_train = []
    transform_train = []
    
    if hflip:
        transform_train.append(transforms.RandomHorizontalFlip())
    
    transform_train.append(transforms.ToTensor())
    transform_test.append(transforms.ToTensor())
    
    if center:
        transform_train.append(transforms.Lambda(lambda x: x - x.mean()))
        transform_test.append(transforms.Lambda(lambda x: x - x.mean()))
    
    transform_train = transforms.Compose(transform_train)
    transform_test = transforms.Compose(transform_test)
        
    
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(root='./data/manual/%s/tv_train' % name, transform=transform_train), 
        batch_size=128, 
        shuffle=True, 
        num_workers=8,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(root='./data/manual/%s/tv_val' % name, transform=transform_test), 
        batch_size=256, 
        shuffle=False, 
        num_workers=8,
        pin_memory=True,
    )
    
    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(root='./data/manual/%s/test' % name, transform=transform_test), 
        batch_size=256, 
        shuffle=False, 
        num_workers=8,
        pin_memory=True,
    )
    
    n_train_batches = len(train_loader)
    n_val_batches = len(val_loader)
    n_test_batches = len(test_loader)
    
    return {
        'train_loader'    : train_loader, 
        'val_loader'      : val_loader,
        'test_loader'     : test_loader, 
        'n_train_batches' : n_train_batches,
        'n_val_batches'   : n_val_batches,
        'n_test_batches'  : n_test_batches,
        'num_classes' : len(train_loader.dataset.classes),
    }

def MNIST():
    return basic('MNIST', center=False, hflip=False)


def fashionMNIST():
    return basic('fashionMNIST', center=False, hflip=True)


def SVHN():
    return basic('SVHN', center=True, hflip=False)


def STL10():
    return basic('STL10', center=True, hflip=True)


def CIFAR(name='CIFAR10'):
    
    mn = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*mn),
    ])
        
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*mn),
    ])
    
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(root='./data/manual/%s/tv_train' % name, transform=transform_train), 
        batch_size=128, 
        shuffle=True, 
        num_workers=8,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(root='./data/manual/%s/tv_val' % name, transform=transform_test), 
        batch_size=256, 
        shuffle=False, 
        num_workers=8,
        pin_memory=True,
    )
    
    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(root='./data/manual/%s/test' % name, transform=transform_test), 
        batch_size=256, 
        shuffle=False, 
        num_workers=8,
        pin_memory=True,
    )
    
    n_train_batches = len(train_loader)
    n_val_batches = len(val_loader)
    n_test_batches = len(test_loader)
    
    return {
        'train_loader'    : train_loader, 
        'val_loader'      : val_loader,
        'test_loader'     : test_loader, 
        'n_train_batches' : n_train_batches,
        'n_val_batches'   : n_val_batches,
        'n_test_batches'  : n_test_batches,
        'num_classes' : len(train_loader.dataset.classes),
    }

