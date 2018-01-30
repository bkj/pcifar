#!/usr/bin/env python

"""
    main.py
"""

from __future__ import division, print_function

import os
import sys
import json
import argparse
import numpy as np
from time import time
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms

from models import *
from lr import LRSchedule
from utils import progress_bar

cudnn.benchmark = True

# --
# Params

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, default='resnet18')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr-schedule', type=str, default='linear')
    parser.add_argument('--lr-init', type=float, default=0.1)
    
    parser.add_argument('--train-size', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--reduce-p-survive', action="store_true")
    
    args = parser.parse_args()
    args.model_name = '%s-%s-%d' % (args.net.lower(), args.lr_schedule, args.epochs)
    return args

args = parse_args()

nets = {
    'vgg' : VGG,
    'resnet18' : ResNet18,
    'stochastic_resnet18' : StochasticResNet18,
    'deadnet18' : DeadNet18,
    'resnet34' : ResNet34,
    'nonet34' : NoNet34,
    'deadnet34' : DeadNet34,
    'googlenet' : GoogLeNet,
    'densenet121' : DenseNet121,
    'resnext29_2x64d' : ResNeXt29_2x64d,
    'mobilenet' : MobileNet,
    'dpn92' : DPN92,
    'shufflenetg2' : ShuffleNetG2,
}

# --
# IO

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)

if args.train_size < 1:
    train_inds, val_inds = train_test_split(np.arange(len(trainset)), train_size=args.train_size, random_state=args.seed + 1)
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=128,
        num_workers=8,
        pin_memory=True,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(train_inds),
    )
    
    valloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=256,
        num_workers=8,
        pin_memory=True,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(val_inds),
    )
else:
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=128,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )


testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=256,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
)

# --
# Helpers

def do_eval(epoch, dataloader):
    _ = net.eval()
    test_loss, correct, total = 0, 0, 0
    for batch_idx, (data, targets) in enumerate(dataloader):
        data, targets = Variable(data, volatile=True).cuda(), Variable(targets).cuda()
        outputs = net(data)
        loss = F.cross_entropy(outputs, targets)
        
        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        
        progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    return float(correct) / total


lr_scheduler = getattr(LRSchedule, args.lr_schedule)(lr_init=args.lr_init, epochs=args.epochs)

# --
# Define model

net = nets[args.net]().cuda()
print(net, file=sys.stderr)

optimizer = optim.SGD(net.parameters(), lr=lr_scheduler(0), momentum=0.9, weight_decay=5e-4)

# --
# Train

for epoch in range(0, args.epochs):
    print("Epoch=%d" % epoch, file=sys.stderr)
    
    if args.reduce_p_survive:
        if args.net == 'stochastic_resnet18':
            net.set_p_survive([1.0, 0.7, 0.7, 0.7])
    
    _ = net.train()
    train_loss, correct, total = 0, 0, 0
    batches_per_epoch = len(trainloader)
    for batch_idx, (data, targets) in enumerate(trainloader):
        data, targets = Variable(data).cuda(), Variable(targets).cuda()
        
        print(lr_scheduler(epoch + batch_idx / batches_per_epoch), file=sys.stderr)
        LRSchedule.set_lr(optimizer, lr_scheduler(epoch + batch_idx / batches_per_epoch))
        
        optimizer.zero_grad()
        outputs = net(data)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.data[0]
        predicted = torch.max(outputs.data, 1)[1]
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        
        progress_bar(batch_idx, batches_per_epoch, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    train_acc = float(correct) / total
    val_acc   = do_eval(epoch, valloader) if args.train_size < 1 else None
    test_acc  = do_eval(epoch, testloader)
    
    print(json.dumps({
        'epoch'     : epoch,
        'train_acc' : train_acc,
        'val_acc'   : val_acc,
        'test_acc'  : test_acc,
    }))


if not os.path.exists('./results/states'):
    _ = os.makedirs('./results/states')

model_path = os.path.join('results', 'states', args.model_name)
print('saving model: %s' % model_path, file=sys.stderr)
torch.save(net.state_dict(), model_path)
