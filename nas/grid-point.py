#!/usr/bin/env python

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
from collections import OrderedDict

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

# --

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--hot-start', action="store_true")
    
    parser.add_argument('--dataset', type=str, default='CIFAR10')
    
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr-schedule', type=str, default='linear')
    parser.add_argument('--lr-init', type=float, default=0.1)
    
    parser.add_argument('--lr-fail-factor', type=float, default=0.5)
    parser.add_argument('--max-failures', type=int, default=3)
    
    parser.add_argument('--run', type=str, default='0')
    parser.add_argument('--train-history', action='store_true')
    
    args = parser.parse_args()
    
    config = args.config
    del args.config
    return args, config

args, config = parse_args()
print >> sys.stderr, 'grid-point.py: starting'

# --
# Params

# Set outdir
file_prefix = os.path.join('results', args.run)

# Set model architecture
if not config:
    print >> sys.stderr, 'sampling config'
    config = sample_config()
    config_path = os.path.join(file_prefix, 'configs', config['model_name'])
    open(config_path, 'w').write(json.dumps(config))
else:
    print >> sys.stderr, 'loading config'
    config = json.load(open(config))

config.update({'args' : vars(args)})
print >> sys.stderr, 'config: %s' % json.dumps(config)

# Set dataset
if args.dataset == 'CIFAR10':
    ds = CIFAR(name='CIFAR10')
elif args.dataset == 'CIFAR100':
    ds = CIFAR(name='CIFAR100')
else:
    raise Exception('!! unknown dataset')

# Set output
p = os.path.join(file_prefix, 'configs')
if not os.path.exists(p):
    _ = os.makedirs(p)

for p in ['states', 'hists']:
    p = os.path.join(file_prefix, args.dataset, p)
    if not os.path.exists(p):
        _ = os.makedirs(p)

hist_path = os.path.join(file_prefix, args.dataset, 'hists', config['model_name'])
model_path = os.path.join(file_prefix, args.dataset, 'states', config['model_name'])

# Set LR schedule
lr_schedule = getattr(LRSchedule, args.lr_schedule)
lr_schedule = functools.partial(lr_schedule, lr_init=args.lr_init, epochs=args.epochs)

# --
# Training helpers

def train_epoch(net, loader, opt, epoch, n_train_batches=ds['n_train_batches'], train_history=False):
    _ = net.train()
    all_loss, correct, total = 0, 0, 0
    history = []
    gen = tqdm(enumerate(loader), total=len(loader))
    for batch_idx, (data, targets) in gen:
        LRSchedule.set_lr(opt, lr_schedule(epoch + batch_idx / n_train_batches))
        
        data, targets = Variable(data.cuda()), Variable(targets.cuda())
        outputs, loss = net.train_step(data, targets, opt)
        
        if np.isnan(loss):
            return np.nan, np.nan, []
        
        predicted = outputs.data.max(1)[1]
        batch_correct = (predicted == targets.data).cpu().sum()
        batch_size = targets.size(0)
        
        if train_history:
            history.append((loss, batch_correct / batch_size))
        
        total += batch_size
        correct += batch_correct
        all_loss += loss
        
        curr_acc = correct / total
        curr_loss = all_loss / (batch_idx + 1)
        gen.set_postfix(OrderedDict([('epoch', epoch), ('train_loss', curr_loss), ('train_acc', curr_acc)]))
    
    return curr_acc, curr_loss, history


def eval(net, epoch, loader, mode='val'):
    _ = net.eval()
    all_loss, correct, total = 0, 0, 0
    gen = tqdm(enumerate(loader), total=len(loader))
    for batch_idx, (data, targets) in gen:
        
        data, targets = Variable(data.cuda(), volatile=True), Variable(targets.cuda())
        outputs = net(data)
        loss = F.cross_entropy(outputs, targets).data[0]
        
        predicted = outputs.data.max(1)[1]
        batch_correct = (predicted == targets.data).cpu().sum()
        batch_size = targets.size(0)
        
        total += batch_size
        correct += batch_correct
        all_loss += loss
                
        curr_acc = correct / total
        curr_loss = all_loss / (batch_idx + 1)
        gen.set_postfix(OrderedDict([('epoch', epoch), ('%s_loss' % mode, curr_loss), ('%s_acc' % mode, curr_acc)]))
    
    return curr_acc, curr_loss


# --
# Train

net = RNet(
    config['op_keys'],
    config['red_op_keys'],
    num_classes=ds['num_classes']
).cuda() 

print >> sys.stderr, net

if args.hot_start:
    epoch = len(open(hist_path).read().splitlines())
    print >> sys.stderr, 'hot start at epoch %d' % epoch
    net.load_state_dict(torch.load(model_path))
else:
    epoch = 0

opt = optim.SGD(net.parameters(), lr=lr_schedule(float(epoch)), momentum=0.9, weight_decay=5e-4)

fail_counter = 0
histfile = open(hist_path, 'a')
while epoch < args.epochs:

    # Train
    train_acc, train_loss, train_history =\
        train_epoch(net, ds['train_loader'], opt, epoch, train_history=args.train_history)
    
    # Sometimes the models don't converge...
    # This is just a heuristic, may not lead to the fairest comparisons
    if np.isnan(train_loss):
        fail_counter += 1
        if fail_counter <= args.max_failures:
            print >> sys.stderr, 'grid-point.py: train_loss is NaN -- reducing LR and restarting'
            args.lr_init *= args.lr_fail_factor
            lr_schedule = functools.partial(lr_schedule, lr_init=args.lr_init, epochs=args.epochs)
            net = RNet(config['op_keys'], config['red_op_keys']).cuda() 
            opt = optim.SGD(net.parameters(), lr=lr_schedule(0.0), momentum=0.9, weight_decay=5e-4)
            epoch = 0
            continue
        else:
            print >> sys.stderr, 'grid-point.py: train_loss is NaN -- too many failures -- exiting'
            os._exit(0)
    
    
    # Eval
    val_acc, val_loss = eval(net, epoch, ds['val_loader'], mode='val')
    test_acc, test_loss = eval(net, epoch, ds['test_loader'], mode='test')
    
    # Log
    histfile.write(json.dumps({
        'epoch'              : epoch, 
        'lr'                 : lr_schedule(epoch + 1),
        
        'train_acc'          : train_acc, 
        'train_loss'         : train_loss,
        'train_history'      : train_history,
        
        'val_acc'            : val_acc, 
        'val_loss'           : val_loss,
        
        'test_acc'           : test_acc,
        'test_loss'          : test_loss,
    }) + '\n')
    histfile.flush()
    
    epoch += 1
    print

# --
# Save + close

torch.save(net.state_dict(), model_path)
histfile.close()