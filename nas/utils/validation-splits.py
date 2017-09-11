#!/usr/bin/env python

"""
    validation-splits.py
    
    Symlink 10% of each dataset as a validation split
"""

import os
import sys
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split

train_size = 0.9
random_state = 123

names = ['MNIST', 'fashionMNIST', 'STL10', 'CIFAR10', 'CIFAR100', 'SVHN']

for name in names:
    root = os.path.abspath(os.path.join('data', 'manual', name))
    print >> sys.stderr, 'validation-splits.py: %s' % root
    
    src = os.path.join(root, 'train')
    dests = map(lambda x: os.path.join(root, x), ['tv_train', 'tv_val'])
    
    fs = map(lambda x: x.split('/')[-2:], glob(os.path.join(src, '*', '*')))
    
    df = pd.DataFrame(fs)
    df.columns = ('lab', 'fname')
    df['src'] = df.apply(lambda x: os.path.join(src, x['lab'], x['fname']), 1)
    
    train, val = train_test_split(df, stratify=df.lab, train_size=train_size, random_state=random_state)
    train, val = train.copy(), val.copy()
    
    for dest, split in zip(dests, [train, val]):
        split['dest'] = split.apply(lambda x: os.path.join(dest, x['lab'], x['fname']), 1)
        
        # Create directories
        for d in split.dest.apply(os.path.dirname).unique():
            if not os.path.exists(d):
                os.makedirs(d)
                
        _ = split[['src', 'dest']].apply(lambda x: os.symlink(x['src'], x['dest']), 1)


