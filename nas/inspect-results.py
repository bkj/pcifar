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

# --
# IO

exp_dir = './results/grid/0/'

# Load configs
configs = []
config_paths = glob(os.path.join(exp_dir, 'configs', '*'))
for config_path in config_paths:
    configs.append(json.load(open(config_path)))

configs = pd.DataFrame(configs)
configs = configs[['model_name', 'op_keys', 'red_op_keys', 'args']]

# Load histories
hists = []
hist_paths = glob(os.path.join(exp_dir, 'hists', '*'))
for hist_path in hist_paths:
    hist = map(json.loads, open(hist_path))
    for h in hist:
        h.update({'model_name' : os.path.basename(hist_path)})
    hists += hist

hists = pd.DataFrame(hists)
hists = hists[['model_name', 'epoch', 'train_acc', 'test_acc']]

# --
# Cleaning

# !! Some nontrivial number of these appear to diverge
# !! Need to figure out why
hists.model_name.unique().shape
hists[hists.test_acc == 0.1].model_name.unique().shape

sub = hists[hists.epoch == hists.epoch.max()]
bad_models = sub.model_name[sub.train_acc == 0.1].unique()

configs = configs[~configs.model_name.isin(bad_models)]
hists = hists[~hists.model_name.isin(bad_models)]

# --
# Enrich

# Add accuracy at last iteration
z = hists.groupby('model_name').test_acc.apply(lambda x: x.tail(1)).reset_index()
del z['level_1']
configs = pd.merge(configs, z)

# --
# Plot all accuracy curves

_ = hists.groupby('model_name').test_acc.apply(lambda x: plt.plot(x.reset_index(drop=True), alpha=0.25))
show_plot()


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

