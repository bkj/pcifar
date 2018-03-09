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
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms

from models import *
from lr import LRSchedule
from utils import progress_bar, set_seeds

torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True

from basenet.helpers import to_numpy

# # >>
# sys.path.append('/home/bjohnson/projects/pipenet')
# from pipenet import PipeNet

# sys.path.append('/home/bjohnson/projects/ripenet/workers')
# from cell_worker import CellWorker
# # <<

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
    args.model_name = '%s-%s-%d-%f' % (args.net.lower(), args.lr_schedule, args.epochs, args.train_size)
    return args

args = parse_args()
set_seeds(args.seed)

nets = {
    # 'vgg'      : VGG,
    'resnet18' : ResNet18,
    # 'stochastic_resnet18' : StochasticResNet18,
    # 'deadnet18'       : DeadNet18,
    # 'resnet34'        : ResNet34,
    # 'nonet34'         : NoNet34,
    # 'deadnet34'       : DeadNet34,
    # 'googlenet'       : GoogLeNet,
    # 'densenet121'     : DenseNet121,
    # 'resnext29_2x64d' : ResNeXt29_2x64d,
    # 'mobilenet'       : MobileNet,
    # 'dpn92'           : DPN92,
    # 'shufflenetg2'    : ShuffleNetG2,
    # 'pipenet'         : PipeNet,
    # 'cell_worker'     : CellWorker, 
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
    train_inds, val_inds = train_test_split(np.arange(len(trainset)), train_size=args.train_size, random_state=1111)
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
        
        test_loss += to_numpy(loss.data[0])
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += to_numpy(predicted.eq(targets.data).cpu().sum())
        
        progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    return float(correct) / total


lr_scheduler = getattr(LRSchedule, args.lr_schedule)(lr_init=args.lr_init, epochs=args.epochs)

# --
# Define model

net = nets[args.net]().cuda()

optimizer = optim.SGD(net.parameters(), lr=lr_scheduler(0), momentum=0.9, weight_decay=5e-4)

if args.net == 'pipenet':
    print('setting pipes', file=sys.stderr)
    net.reset_pipes()

print(net, file=sys.stderr)

# --
# Train

for epoch in range(args.epochs):
    print("Epoch=%d" % epoch, file=sys.stderr)
    
    _ = net.train()
    train_loss, correct, total = 0, 0, 0
    batches_per_epoch = len(trainloader)
    for batch_idx, (data, targets) in enumerate(trainloader):
        data, targets = Variable(data).cuda(), Variable(targets).cuda()
        
        # LRSchedule.set_lr(optimizer, lr_scheduler(epoch + batch_idx / batches_per_epoch))
        
        optimizer.zero_grad()
        outputs = net(data)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()
        
        # train_loss += to_numpy(loss.data[0])
        # predicted  = torch.max(outputs.data, 1)[1]
        # total      += targets.shape[0]
        # correct    += to_numpy(predicted.eq(targets.data).cpu().sum())
        
        progress_bar(batch_idx, batches_per_epoch)# , 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            # % (to_numpy(train_loss)/(batch_idx+1), 100.*correct/total, correct, total))
    
    print(json.dumps({
        'epoch'     : epoch,
        'train_acc' : float(correct) / total,
        'val_acc'   : do_eval(epoch, valloader) if args.train_size < 1 else None,
        'test_acc'  : do_eval(epoch, testloader),
        'lr'        : lr_scheduler(epoch + batch_idx / batches_per_epoch),
    }))


if not os.path.exists('./results/states'):
    _ = os.makedirs('./results/states')

model_path = os.path.join('results', 'states', args.model_name)
print('saving model: %s' % model_path, file=sys.stderr)
torch.save(net.state_dict(), model_path)
