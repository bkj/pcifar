#!/usr/bin/env python

"""
    inspect-results.py
"""

import os
import sys
from tqdm import tqdm

import numpy as np
import pandas as pd
import ujson as json
from glob import glob

from rsub import *
from matplotlib import pyplot as plt

pd.set_option('display.width', 200)

# --
# IO

exp_dir = './results/1/'

# Load configs
configs = []
config_paths = glob(os.path.join(exp_dir, 'configs', '*'))
for config_path in config_paths:
    config = json.load(open(config_path))
    config.update(config['args'])
    del config['args']
    configs.append(config)

assert len(configs) > 0
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

lin_models = configs.model_name[configs.lr_schedule == 'linear'].unique()
hists['linear'] = hists.model_name.isin(lin_models)

# --
# Validation accuracy vs. test accuracy -- are they close?

sub = hists[hists.epoch == 19]
_ = plt.scatter(sub.test_acc, sub.val_acc, s=3, alpha=0.25)
_ = plt.xlim(0.75, 1)
_ = plt.ylim(0.75, 1)
show_plot()

# Yes

# --
# Performance over time

_ = hists[hists.linear].groupby('model_name').test_acc.apply(lambda x: plt.plot(x.reset_index(drop=True), alpha=0.25))
show_plot()

_ = hists[~hists.linear].groupby('model_name').test_acc.apply(lambda x: plt.plot(x.reset_index(drop=True), alpha=0.25))
show_plot()

# >>
sub = hists[~hists.linear]
sub['test_acc_adj'] = sub.groupby('model_name').val_acc.apply(lambda x: x - x.iloc[-1])
sub['test_acc_adj_cent'] = sub.groupby('epoch').test_acc_adj.apply(lambda x: x - x.mean())

_ = sub.groupby('model_name').test_acc_adj_cent.apply(lambda x: plt.plot(x.reset_index(drop=True), alpha=0.25))
show_plot()

all_res = []
sub = hists[~hists.linear]
for _ in tqdm(range(100)):
    sel = sub.model_name.sample(100, replace=False)
    z = sub[sub.model_name.isin(sel)]
    top5 = set(z[z.epoch == 19].sort_values('val_acc').tail(10).model_name)
    
    res = []
    for i in range(20):
        tmp = set(z[z.epoch == i].sort_values('val_acc').tail(10).model_name)
        res.append(len(tmp.intersection(top5)))
    
    all_res.append(res)

all_res = np.vstack(all_res)

_ = plt.plot(np.mean(all_res, axis=0))
_ = plt.plot(np.mean(all_res_lin, axis=0))
show_plot()

# <<

# --
# Save as numpy arrays
np.save('./results/1/lin_train_acc', np.vstack(hists[hists.linear].groupby('model_name').train_acc.apply(np.array)))
np.save('./results/1/lin_val_acc', np.vstack(hists[hists.linear].groupby('model_name').val_acc.apply(np.array)))
np.save('./results/1/lin_test_acc', np.vstack(hists[hists.linear].groupby('model_name').test_acc.apply(np.array)))

np.save('./results/1/cyc_train_acc', np.vstack(hists[~hists.linear].groupby('model_name').train_acc.apply(np.array)))
np.save('./results/1/cyc_val_acc', np.vstack(hists[~hists.linear].groupby('model_name').val_acc.apply(np.array)))
np.save('./results/1/cyc_test_acc', np.vstack(hists[~hists.linear].groupby('model_name').test_acc.apply(np.array)))

best_lin = hists[hists.linear].sort_values('val_acc', ascending=False).model_name.head()
best_cyc = hists[~hists.linear].sort_values('val_acc', ascending=False).model_name.head()

configs[configs.model_name.isin(best_lin)]
configs[configs.model_name.isin(best_cyc)]



