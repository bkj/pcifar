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
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms

from nas import RNet, sample_config
from data import *
from lr import LRSchedule

cudnn.benchmark = True

run = 0
file_prefix = os.path.join('results', 'grid', str(run))

for p in ['states', 'configs', 'hists']:
    p = os.path.join(file_prefix, p)
    if not os.path.exists(p):
        _ = os.makedirs(p)

print >> sys.stderr, 'grid-point.py: starting'

# --
# Params

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr-schedule', type=str, default='linear')
    parser.add_argument('--lr-init', type=float, default=0.1)
    return parser.parse_args()

args = parse_args()

# Set learning rate schedule
lr_schedule = getattr(LRSchedule, args.lr_schedule)
lr_schedule = functools.partial(lr_schedule, lr_init=args.lr_init, epochs=args.epochs)

# Set dataset
ds = CIFAR10()

# Set model architecture
config = sample_config()
config.update({'args' : vars(args)})
print >> sys.stderr, json.dumps(config)

config_path = os.path.join(file_prefix, 'configs', config['model_name'])
open(config_path, 'w').write(json.dumps(config))


# --
# Training helpers

def train_epoch(net, loader, opt, epoch, n_train_batches=ds['n_train_batches']):
    _ = net.train()
    all_loss = 0
    correct = 0
    total = 0
    gen = tqdm(enumerate(loader), total=len(loader))
    for batch_idx, (data, targets) in gen:
        
        # !! Is this slow?
        LRSchedule.set_lr(opt, lr_schedule(epoch + batch_idx / n_train_batches))
        
        data, targets = Variable(data.cuda()), Variable(targets.cuda())
        outputs, loss = net.train_step(data, targets, opt)
        all_loss += loss
        
        predicted = outputs.data.max(1)[1]
        total += targets.size(0)
        correct += (predicted == targets.data).cpu().sum()
        
        # !! Is this slow?
        curr_acc = correct / total
        curr_loss = all_loss / (batch_idx + 1)
        gen.set_postfix({'curr_acc' : curr_acc, 'curr_loss' : curr_loss})
    
    if np.isnan(curr_loss):
        print >> sys.stderr, 'grid-point.py: train_loss is NaN -- exiting'
        os._exit(0)
    
    return curr_acc, curr_loss


def eval(net, epoch, loader):
    _ = net.eval()
    all_loss = 0
    correct = 0
    total = 0
    gen = tqdm(enumerate(loader), total=len(loader))
    for batch_idx, (data, targets) in gen:
        
        data, targets = Variable(data.cuda(), volatile=True), Variable(targets.cuda())
        
        outputs = net(data)
        loss = F.cross_entropy(outputs, targets).data[0]
        
        all_loss += loss
        
        predicted = outputs.data.max(1)[1]
        total += targets.size(0)
        correct += (predicted == targets.data).cpu().sum()
        
        # !! Is this slow?
        curr_acc = correct / total
        curr_loss = all_loss / (batch_idx + 1)
        gen.set_postfix({'curr_acc' : curr_acc, 'curr_loss' : curr_loss})

    
    return curr_acc, curr_loss

# --
# Train

net = RNet(config['op_keys'], config['red_op_keys']).cuda() 
opt = optim.SGD(net.parameters(), lr=lr_schedule(0.0), momentum=0.9, weight_decay=5e-4)
print >> sys.stderr, net

hist_path = os.path.join(file_prefix, 'hists', config['model_name'])
hist = open(hist_path, 'w')

for epoch in range(args.epochs):
    print >> sys.stderr, "Epoch=%d" % epoch
    
    print >> sys.stderr, "train"
    train_acc, train_loss = train_epoch(net, ds['train_loader'], opt, epoch)
    
    print >> sys.stderr, "val"
    val_acc, val_loss = eval(net, epoch, ds['val_loader'])
    
    print >> sys.stderr, "test"
    test_acc, test_loss = eval(net, epoch, ds['test_loader'])
    
    # Log
    hist.write(json.dumps({
        'epoch'      : epoch, 
        'train_acc'  : train_acc, 
        'train_loss' : train_loss,
        'val_acc'    : val_acc, 
        'val_loss'   : val_loss,
        'test_acc'   : test_acc,
        'test_loss'  : test_loss,
    }) + '\n')
    hist.flush()

# --
# Save final model

model_path = os.path.join(file_prefix, 'states', config['model_name'])
print >> sys.stderr, 'saving model: %s' % model_path
torch.save(net.state_dict(), model_path)
