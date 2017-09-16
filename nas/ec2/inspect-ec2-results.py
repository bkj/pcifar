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

# --

root = './results/ec2/CIFAR10/hists/'

hists = []
hist_paths = glob(os.path.join(root, '*'))
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

from scipy.stats import spearmanr

f = np.array(hists[hists.epoch == 19].val_acc)
i = np.array(hists[hists.epoch == 1].val_acc)

sel_a = np.random.choice(f.shape[0], 100000)
sel_b = np.random.choice(f.shape[0], 100000)

((f[sel_a] > f[sel_b]) == (i[sel_a] > i[sel_b])).mean()




# --

# --
# Plot

_ = hists.groupby('model_name').test_acc.apply(lambda x: plt.plot(x.reset_index(drop=True), alpha=0.25))
show_plot()

# --
# Model selection w/ hyperband (using last seen metric)

X_orig = np.vstack(hists.groupby('model_name').val_acc.apply(np.array)).T

R = 2
alpha = 0.5
X = X_orig.copy()

popsize = X.shape[1]

pers = np.arange(R, 20, R)

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

# model in top 1%, using 20% of the computation

# --
# Model selection w/ hyperband (using earlier stopping)

X = X_orig.copy()
# train_sel = np.random.choice(X.shape[1], 50)
# train = X[:,train_sel].T

from sklearn.svm import SVR
from sklearn.decomposition import PCA, TruncatedSVD

X = X[:,X[-1] > 0.6]

svd = TruncatedSVD(n_components=9)
z = svd.fit_transform(X[:10].T)

np.corrcoef(z[:,0], X.T[:,-1])[0,1]
np.corrcoef(X.T[:,9], X.T[:,-1])[0,1]







