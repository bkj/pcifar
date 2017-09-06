#!/usr/bin/env python

"""
    nas-main.py
"""

from __future__ import division

import os
import sys
import json
import functools
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms

from nas import *
from lr import LRSchedule

sys.path.append('..')
from utils import progress_bar

cudnn.benchmark = True

# --
# Params

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, default='nas')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr-schedule', type=str, default='linear')
    parser.add_argument('--lr-init', type=float, default=0.1)
    parser.add_argument('--model-name', type=str)
    args = parser.parse_args()
    if not args.model_name:
        args.model_name = '%s-%d-%s-%0.2f' % (args.net.lower(), args.epochs, args.lr_schedule, args.lr_init)
        print >> sys.stderr, args.model_name
    
    return args

args = parse_args()

lr_schedule = getattr(LRSchedule, args.lr_schedule)
lr_schedule = functools.partial(lr_schedule, lr_init=args.lr_init, epochs=args.epochs)

# --
# IO

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), # !! ??
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), # !! ??
])

trainset = torchvision.datasets.CIFAR10(root='./data/CIFAR10', train=True, download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, 
    batch_size=128, 
    shuffle=True, 
    num_workers=8,
    pin_memory=True
)

testset = torchvision.datasets.CIFAR10(root='./data/CIFAR10', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, 
    batch_size=256, 
    shuffle=False, 
    num_workers=8,
    pin_memory=True,
)

# --
# Helpers

def test(epoch, testloader):
    _ = net.eval()
    all_loss = 0
    correct = 0
    total = 0
    for batch_idx, (data, targets) in enumerate(testloader):
        
        data, targets = Variable(data.cuda(), volatile=True), Variable(targets.cuda())
        
        outputs = net(data)
        loss = F.cross_entropy(outputs, targets).data[0]
        
        all_loss += loss
        
        predicted = outputs.data.max(1)[1]
        total += targets.size(0)
        correct += (predicted == targets.data).cpu().sum()
        
        # !! Is this slow?
        progress_bar(
            batch_idx, 
            len(testloader), 
            'Loss: %.3f | Acc: %.3f (%d/%d)'% (
                all_loss / (batch_idx + 1), 
                float(correct / total), 
                correct, 
                total
            )
        )
    
    return float(correct) / total


# --
# Define model

outfile = open(os.path.join('results', args.model_name + '.jl'), 'w')

batches_per_epoch = len(trainloader)

# ResNet18
op_keys = ('double_bnconv_3', 'identity', 'add')
red_op_keys = ('double_bnconv_3', 'conv_1', 'add')
net = RNet(op_keys, red_op_keys).cuda() 
print >> sys.stderr, net

optimizer = optim.SGD(net.parameters(), lr=lr_schedule(0.0), momentum=0.9, weight_decay=5e-4)

train_accs, test_accs = [], []
for epoch in range(args.epochs):
    print >> sys.stderr, "Epoch=%d" % epoch
    _ = net.train()
    all_loss = 0
    correct = 0
    total = 0
    for batch_idx, (data, targets) in enumerate(trainloader):
        
        # !! Is this slow?
        LRSchedule.set_lr(optimizer, lr_schedule(epoch + batch_idx / batches_per_epoch))
        
        data, targets = Variable(data.cuda()), Variable(targets.cuda())
        outputs, loss = net.train_step(data, targets, optimizer)
        all_loss += loss
        
        predicted = outputs.data.max(1)[1]
        total += targets.size(0)
        correct += (predicted == targets.data).cpu().sum()
        
        # !! Is this slow?
        progress_bar(
            batch_idx, 
            len(trainloader), 
            'Loss: %.3f | Acc: %.3f (%d/%d)'% (
                all_loss / (batch_idx + 1), 
                float(correct) / total, 
                correct, 
                total
            )
        )
    
    train_accs.append(float(correct) / total)
    test_accs.append(test(epoch, testloader))
    
    outfile.write(json.dumps({'epoch' : epoch, 'train_acc' : train_accs[-1], 'test_acc' : test_accs[-1]}) + '\n')
    outfile.flush()

if not os.path.exists('./results/states'):
    _ = os.makedirs('./results/states')

model_path = os.path.join('results', 'states', args.model_name)
print >> sys.stderr, 'saving model: %s' % model_path
torch.save(net.state_dict(), model_path)
