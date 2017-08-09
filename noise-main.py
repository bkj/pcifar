#!/usr/bin/env python

"""
    main.py
"""

import os
import sys
import json
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

from models import *
from utils import progress_bar

# --
# Params

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, default='resnet18')
    parser.add_argument('--model-name', type=str)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr-schedule', type=str, default='linear')
    args = parser.parse_args()
    if not args.model_name:
        args.model_name = '%s-%s-%d' % (args.net.lower(), args.lr_schedule, args.epochs)
    return args

args = parse_args()

nets = {
    'vgg' : VGG,
    'resnet18' : ResNet18,
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
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), # !! ??
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), # !! ??
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, 
    batch_size=128, 
    shuffle=True, 
    num_workers=8,
    pin_memory=True
)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, 
    batch_size=256, 
    shuffle=False, 
    num_workers=8,
    pin_memory=True,
)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# --
# Helpers

# data, targets = next(iter(trainloader))

def fix_data(data, targets):
    # !! Want to add ability to chance prevalance of noise
    noise_sel = (targets != 0) & (targets != 5)
    targets[targets == 0] = 0
    targets[targets == 5] = 1
    targets[noise_sel] = torch.LongTensor(np.random.choice((0, 1), noise_sel.sum()))
    
    # sel = torch.nonzero((targets == 0) | (targets == 5)).squeeze()
    # targets = targets[sel]
    # targets[targets == 0] = 0
    # targets[targets == 5] = 1
    # data = data[sel]
    return data, targets
    

def train(epoch):
    print >> sys.stderr, "Epoch=%d" % epoch
    _ = net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (data, targets) in enumerate(trainloader):
        
        # <<
        data, targets = fix_data(data, targets)
        # >>
        
        data, targets = Variable(data.cuda()), Variable(targets.cuda())
        
        optimizer.zero_grad()
        outputs = net(data)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    return float(correct) / total


def test(epoch):
    _ = net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = Variable(inputs.cuda(), volatile=True), Variable(targets.cuda())
        outputs = net(inputs)
        # loss = F.cross_entropy(outputs, targets)
        
        # test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        
        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (1.0/(batch_idx+1), 100.*correct/total, correct, total))
    
    return float(correct) / total

# --
# Learning rate scheduling

def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def lr_schedule_step(epoch, breaks=(150, 250)):
    if epoch < breaks[0]:
        return 0.1
    elif epoch < breaks[1]:
        return 0.01
    else:
        return 0.001

def lr_schedule_linear(epoch, lr_init=0.1, epochs=args.epochs):
    return lr_init * float(epochs - epoch) / epochs

lr_schedules = {
    "step" : lr_schedule_step,
    "linear" : lr_schedule_linear,
}

lr_schedule = lr_schedules[args.lr_schedule]

# --
# Define model

net = nets[args.net]().cuda()
print >> sys.stderr, net
cudnn.benchmark = True

optimizer = optim.SGD(net.parameters(), lr=lr_schedule(0), momentum=0.9, weight_decay=5e-4)

train_accs, test_accs = [], []
for epoch in range(0, args.epochs):
    # Set learning rate
    set_lr(optimizer, lr_schedule(epoch))
    
    # Run training
    train_acc = train(epoch)
    test_acc = test(epoch)
    train_accs.append(train_acc)
    test_accs.append(test_acc)
    
    print json.dumps({'train_acc' : train_acc, 'test_acc' : test_acc})

if not os.path.exists('./noise-results/states'):
    _ = os.makedirs('./noise-results/states')

model_path = os.path.join('noise-results', 'states', args.model_name)
print >> sys.stderr, 'saving model: %s' % model_path
torch.save(net.state_dict(), model_path)
