#!/usr/bin/env python

"""
    nas-hyperband.py
"""

import sys
import numpy as np
import pandas as pd

from rsub import *
from matplotlib import pyplot as plt

res = []
mode = 'cyc'
X_orig = np.load('./results/1/%s_val_acc.npy' % mode).T

R = 1
alpha = 0.2
X = X_orig.copy()

popsize = X.shape[1]

pers = np.arange(R, 20, R)

for p in pers:
    popsize = int(np.ceil(alpha * popsize))
    X[p:,X[p].argsort()[:-popsize]] = 0

for x in X_orig.T:
    _ = plt.plot(x, c='grey', alpha=0.1)

_ = plt.plot(X_orig.max(axis=1), c='red')
_ = plt.plot(X.max(axis=1), alpha=0.25, c='b')
_ = plt.ylim(0.5, 0.9)
show_plot()

float((X > 0).sum()) / np.prod(X.shape) # x% of the computation
float((X > 0).sum()) / X.shape[0] # y times more than necessary

(X > 0).sum(axis=1)

(X[-1].max() < X_orig[-1]).sum()
X_orig.shape