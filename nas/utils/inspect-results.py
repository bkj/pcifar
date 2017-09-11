#!/usr/bin/env python

"""
    inspect-results.py
"""

import os
import sys

import numpy as np
import pandas as pd
import ujson as json
from glob import glob

from rsub import *
from matplotlib import pyplot as plt

pd.set_option('display.width', 200)

# --
# IO

exp_dir = './results/grid/1/'

# Load configs
configs = []
config_paths = glob(os.path.join(exp_dir, 'configs', '*'))
for config_path in config_paths:
    config = json.load(open(config_path))
    config.update(config['args'])
    del config['args']
    configs.append(config)

configs = pd.DataFrame(configs)
configs = configs[['model_name', 'timestamp', 'epochs', 'lr_init', 'lr_schedule', 'op_keys', 'red_op_keys']]
configs = configs.sort_values('timestamp').reset_index(drop=True)

# Load histories
hists = []
hist_paths = glob(os.path.join(exp_dir, 'hists', '*'))
for hist_path in hist_paths:
    hist = map(json.loads, open(hist_path))
    for h in hist:
        h.update({'model_name' : os.path.basename(hist_path)})
    hists += hist

hists = pd.DataFrame(hists)
hists = hists[[
    'model_name', 'epoch', 
    'train_loss', 'val_loss', 'test_loss',
    'train_acc', 'val_acc', 'test_acc'
]]

# --
# Cleaning

# Some configs don't have hists (probably control+C)
configs = configs[configs.model_name.isin(hists.model_name)]

# Some models diverge and exit early -- drop for now
runtime = hists.groupby('model_name').epoch.max()
keep = runtime[runtime == runtime.max()].index

configs = configs[configs.model_name.isin(keep)].reset_index(drop=True)
hists = hists[hists.model_name.isin(keep)].reset_index(drop=True)

# --
# Validation accuracy vs. test accuracy -- are they close?

sub = hists[hists.epoch == 19]
_ = plt.scatter(sub.test_acc, sub.val_acc, s=3, alpha=0.25)
_ = plt.xlim(0.75, 1)
_ = plt.ylim(0.75, 1)
show_plot()

# Yes, correlated.  Accuracies here are lower than I'm used to, I assume because the training set is lower.
# resnet18 gets val_acc=0.887, test_acc=0.885
# so, after training 135 models, 
# <<

good_names = z.model_name[z.test_acc > 0.9193]
configs[configs.model_name.isin(good_names)].sort_values('test_acc')

# --
# How does rank change over time?
# !! w/ caveat that these are all trained super fast

lin_models = configs.model_name[configs.args.isnull()].unique()
configs['lin'] = configs.model_name.isin(lin_models)
hists['lin'] = hists.model_name.isin(lin_models)

_ = hists[hists.lin].groupby('model_name').test_acc.apply(lambda x: plt.plot(x.reset_index(drop=True), alpha=0.25))
show_plot()

_ = hists[~hists.lin].groupby('model_name').test_acc.apply(lambda x: plt.plot(x.reset_index(drop=True), alpha=0.25))
show_plot()

lin_hists = hists[hists.lin]
cyc_hists = hists[~hists.lin]


sub = lin_hists.copy()
sub = sub[sub.model_name.isin(sub.model_name[(sub.epoch == 19) & (sub.test_acc > 0.6)])]
coefs = [np.corrcoef(sub.test_acc[sub.epoch == i], sub.test_acc[sub.epoch == 19])[0,1] for i in range(20)]
_ = plt.plot(coefs)

sub = cyc_hists.copy()
sub = sub[sub.model_name.isin(sub.model_name[(sub.epoch == 19) & (sub.test_acc > 0.6)])]
coefs = [np.corrcoef(sub.test_acc[sub.epoch == i], sub.test_acc[sub.epoch == 19])[0,1] for i in range(20)]
_ = plt.plot(coefs)
_ = plt.ylim(0.8, 1.0)
show_plot()


# >>

cyc_names = configs.model_name[configs.lr_schedule == 'cyclical']
cyc_hists = hists[hists.model_name.isin(cyc_names)]

train_acc = np.vstack(cyc_hists.groupby('model_name').train_acc.apply(np.array))
val_acc = np.vstack(cyc_hists.groupby('model_name').val_acc.apply(np.array))
test_acc = np.vstack(cyc_hists.groupby('model_name').test_acc.apply(np.array))

np.save('./results/grid/1/train_acc', train_acc)
np.save('./results/grid/1/val_acc', val_acc)
np.save('./results/grid/1/test_acc', test_acc)




