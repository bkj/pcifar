# #!/usr/bin/env python

"""
    grid-point.py
"""

from __future__ import division

import os
import sys
import json
import argparse
import functools
import numpy as np
from hashlib import md5
from datetime import datetime

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

run = 0
file_prefix = os.path.join('results', 'grid', str(run))

for p in ['states', 'configs', 'hists']:
    p = os.path.join(file_prefix, p)
    if not os.path.exists(p):
        _ = os.makedirs(p)

# --
# Params

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr-schedule', type=str, default='linear')
    parser.add_argument('--lr-init', type=float, default=0.1)
    parser.add_argument('--model-name', type=str)
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
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data/CIFAR10', train=True, download=False, transform=transform_train)
train_loader = torch.utils.data.DataLoader(
    trainset, 
    batch_size=128, 
    shuffle=True, 
    num_workers=8,
    pin_memory=True
)

testset = torchvision.datasets.CIFAR10(root='./data/CIFAR10', train=False, download=False, transform=transform_test)
test_loader = torch.utils.data.DataLoader(
    testset, 
    batch_size=256, 
    shuffle=False, 
    num_workers=8,
    pin_memory=True,
)

batches_per_epoch = len(train_loader)

# --
# Helpers

def test(net, epoch, test_loader):
    _ = net.eval()
    all_loss = 0
    correct = 0
    total = 0
    for batch_idx, (data, targets) in enumerate(test_loader):
        
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
            len(test_loader), 
            'Loss: %.3f | Acc: %.3f (%d/%d)'% (
                all_loss / (batch_idx + 1), 
                float(correct / total), 
                correct, 
                total
            )
        )
    
    return float(correct) / total


def train_epoch(net, train_loader, opt, epoch):
    _ = net.train()
    all_loss = 0
    correct = 0
    total = 0
    for batch_idx, (data, targets) in enumerate(train_loader):
        
        # !! Is this slow?
        LRSchedule.set_lr(opt, lr_schedule(epoch + batch_idx / batches_per_epoch))
        
        data, targets = Variable(data.cuda()), Variable(targets.cuda())
        outputs, loss = net.train_step(data, targets, opt)
        all_loss += loss
        
        predicted = outputs.data.max(1)[1]
        total += targets.size(0)
        correct += (predicted == targets.data).cpu().sum()
        
        # !! Is this slow?
        progress_bar(
            batch_idx, 
            len(train_loader), 
            'Loss: %.3f | Acc: %.3f (%d/%d)'% (
                all_loss / (batch_idx + 1), 
                float(correct) / total, 
                correct, 
                total
            )
        )
    
    return float(correct) / total

# --
# Define model

def sample_config():
    config = {
        'args' : vars(args),
        'timestamp' : datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
        'op_keys' : (
            np.random.choice(ops.keys()),
            np.random.choice(ops.keys()),
            # np.random.choice(combs.keys()),
            'add'
        ),
        'red_op_keys' : (
            np.random.choice(red_ops.keys()),
            np.random.choice(red_ops.keys()),
            # np.random.choice(combs.keys()),
            'add'
        ),
    }
    
    # !! Don't want double-pool blocks
    if ('pool' in config['op_keys'][0]) and ('pool' in config['op_keys'][1]):
        print >> sys.stderr, 'resampling...'
        config = sample_config()
    
    config['model_name'] = md5(json.dumps(config)).hexdigest()
    return config


config = sample_config()
print >> sys.stderr, json.dumps(config)

config_path = os.path.join(file_prefix, 'configs', config['model_name'])
open(config_path, 'w').write(json.dumps(config))

# --
# Run training

net = RNet(config['op_keys'], config['red_op_keys']).cuda() 
opt = optim.SGD(net.parameters(), lr=lr_schedule(0.0), momentum=0.9, weight_decay=5e-4)
print >> sys.stderr, net

hist_path = os.path.join(file_prefix, 'hists', config['model_name'])
hist = open(hist_path, 'w')

train_accs, test_accs = [], []
for epoch in range(args.epochs):
    print >> sys.stderr, "Epoch=%d" % epoch
    
    # Train
    train_acc = train_epoch(net, train_loader, opt, epoch)
    train_accs.append(train_acc)
    
    # Eval
    test_acc = test(net, epoch, test_loader)
    test_accs.append(test_acc)
    
    hist.write(json.dumps({
        'epoch' : epoch, 
        'train_acc' : train_acc, 
        'test_acc' : test_acc
    }) + '\n')
    hist.flush()

# --
# Save model

model_path = os.path.join(file_prefix, 'states', config['model_name'])
print >> sys.stderr, 'saving model: %s' % model_path
torch.save(net.state_dict(), model_path)
