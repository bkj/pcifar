#!/usr/bin/env python

"""
    inspect-ec2-results.py
"""

import os
import json
import pandas as pd
import numpy as np
from glob import glob

from rsub import *
from matplotlib import pyplot as plt

pd.set_option('display.width', 200)

root = './results/ec2/'

# --
# Load configs

config_root = os.path.join(root, 'configs')

configs = []
config_paths = glob(os.path.join(config_root, '*'))
for config_path in config_paths:
    config = json.load(open(config_path))
    config.update(config['args'])
    del config['args']
    configs.append(config)

configs = pd.DataFrame(configs)


# --
# Load histories

hist_root = os.path.join(root, 'CIFAR10/hists')

hists = []
hist_paths = glob(os.path.join(hist_root, '*'))
for hist_path in hist_paths:
    hist = map(json.loads, open(hist_path))
    
    model_name = os.path.basename(hist_path)
    for h in hist:
        h.update({'model_name' : model_name})
    
    hists += hist

hists = pd.DataFrame(hists)

hists = hists[[
    'model_name', 'epoch', 'timestamp', 'lr',
    'train_loss', 'val_loss', 'test_loss',
    'train_acc', 'val_acc', 'test_acc'
]]

# --
# Drop failed runs

max_epochs = hists.groupby('model_name').epoch.max()
keep = max_epochs[max_epochs == 19].index
hists = hists[hists.model_name.isin(keep)]

hists = hists.groupby('model_name').apply(lambda x: x.tail(20))
hists = hists.reset_index(drop=True)

assert np.all(hists.model_name.value_counts() == 20)

# --
# Correlation between early and late epochs

# >>
# subset to cyclical LR

configs = configs[configs.lr_schedule == 'cyclical']

hists = hists[hists.model_name.isin(configs.model_name)]
configs = configs[configs.model_name.isin(hists.model_name)]

X_orig = np.vstack(hists.groupby('model_name').val_acc.apply(np.array))
X_orig = X_orig[np.argsort(X_orig[:,0])]

cs = np.linspace(0, 1, X_orig.shape[0])
for c,h in zip(cs, X_orig):
    _ = plt.plot(h, c=plt.cm.rainbow(c), alpha=0.10)

_ = plt.xlim(0, 2)
_ = plt.ylim(0.69, 0.720)
show_plot()

# <<

# >>

from scipy.spatial.distance import pdist, squareform

sub = hists[hists.epoch == 19]
ds = squareform(pdist(np.array(sub.val_acc).reshape(-1, 1), metric='cityblock'))
_ = plt.plot(np.sort(np.hstack(ds)), np.linspace(0, 1, np.prod(ds.shape)))
show_plot()
(ds < 0.006).mean()

# <<


# --
# Plot

_ = hists.groupby('model_name').test_acc.apply(lambda x: plt.plot(x.reset_index(drop=True), alpha=0.25))
show_plot()

# --
# Model selection w/ hyperband (using last seen metric)

X_orig = np.vstack(hists.groupby('model_name').val_acc.apply(np.array)).T

R = 1
alpha = 0.5
X = X_orig.copy()

popsize = X.shape[1]

pers = [1] + list(np.arange(R, 20, R))

for p in pers:
    popsize = int(np.ceil(alpha * popsize))
    X[p:,X[p].argsort()[:-popsize]] = 0

# Plot
for x in X_orig.T:
    _ = plt.plot(x, c='grey', alpha=0.1)

for x in X.T:
    _ = plt.plot(x[x > 0], c='orange', alpha=0.1)

_ = plt.plot(X_orig.max(axis=1), c='red', alpha=0.75, label='frontier')
_ = plt.plot(X_orig[:,X_orig[-1].argmax()], c='green', alpha=0.75, label='best1')
_ = plt.plot(X.max(axis=1), alpha=0.75, c='blue', label='hyperband')
_ = plt.ylim(0.4, 0.9)
_ = plt.legend(loc='lower right')
show_plot()

float((X > 0).sum()) / np.prod(X.shape) # x% of the computation
float((X > 0).sum()) / X.shape[0] # y times more than necessary
(X > 0).sum(axis=1)
(X[-1].max() < X_orig[-1]).mean() 

