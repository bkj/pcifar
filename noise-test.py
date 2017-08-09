import os
import sys
import json
import argparse
import numpy as np

from rsub import *
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms

from models import *
from utils import progress_bar

# --
# IO

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), # !! ??
])

dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_test)
dataloase = torch.utils.data.DataLoader(
    dataset, 
    batch_size=256, 
    shuffle=False, 
    num_workers=8,
    pin_memory=True,
)

def get_features(epoch):
    all_outputs, all_targets = [], []
    for batch_idx, (inputs, targets) in enumerate(dataloase):
        inputs = Variable(inputs.cuda(), volatile=True)
        outputs = net(inputs)
        outputs = outputs.data.cpu().numpy()
        all_outputs.append(outputs)
        all_targets.append(targets.numpy())
        
    return np.vstack(all_outputs), np.hstack(all_targets)

# --
# Load model


net = ResNet18().cuda()
net.load_state_dict(torch.load('./results/states/resnet18-linear-100'))
_ = net.eval()

outputs, targets = get_features(net)
# test_outputs, test_targets = get_features(net)

_ = plt.hist(outputs[targets == 0,0], 100, alpha=0.25, normed=True, log=True)
_ = plt.hist(outputs[targets == 5,0], 100, alpha=0.25, normed=True, log=True)
_ = plt.hist(outputs[(targets != 0) & (targets != 5),0], 100, alpha=0.25, normed=True, log=True)
show_plot()

# --

i = 0.1
o = outputs[:,0]

def f(i):
    fp = ((targets == 0) | (targets == 5))[(o > -i) & (o < i)].mean()
    fn = ((targets != 0) & (targets != 5))[(o < -i) | (o > i)].mean()
    return fp, fn


tmp = np.vstack([f(i) for i in np.arange(0, 2, 0.01)])

_ = plt.plot(tmp[:,0])
_ = plt.plot(tmp[:,1])
show_plot()

i = 1
f(1)
# Can get rid of 97% of the irrelevant labels
# while only getting rid of 5% of the relevant labels
# !! Those 5% might be the hardest 5 percent though

# --
# Ways to train:
#  1) Train on full dataset, w/ uninformative images
#  2) Filter uninformative images, train on results
#    - Filtering procedure: train model on all data, throw out points w/ 
#      - high entropy
#      - high dropout uncertainty
#    then retrain on smaller dataset (faster)

# --

pd.crosstab(outputs[:,0] > outputs[:,5], targets)
# 14, 12 errors
# 2 and 9 errors

# --

from sklearn.svm import LinearSVC

X_train = outputs.copy()
y_train = targets.copy()

f_noise = 0.5
lookup = {
    0 : 0,
    5 : 1
}

y_train = np.array([lookup.get(y, -1) for y in y_train])

n_act = (y_train >= 0).sum()
n_noise = int(n_act * f_noise)

noise_sel = np.random.choice(np.where(y_train < 0)[0], n_noise, replace=False)
act_sel = np.where(y_train >= 0)[0]
sel = np.hstack([noise_sel, act_sel])
sel = np.random.permutation(sel)

X_train, y_train = X_train[sel], y_train[sel]

noise_ind = y_train < 0
y_train[noise_ind] = np.random.choice((0, 1), n_noise)

svc = LinearSVC().fit(X_train, y_train)

preds = svc.decision_function(outputs)
_ = plt.hist(preds[targets == 0], 100, alpha=0.5)
_ = plt.hist(preds[targets == 5], 100, alpha=0.5)
_ = plt.hist(preds[(targets != 0) & (targets != 5)], 100, alpha=0.5)
show_plot()






