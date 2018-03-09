
# !! The networks used in the stochastic depth paper are significantly different
# than these
# 
# Three blocks instead of 4
# Avg. pooling in shortcut instead of conv2d
# (16, 32, 64) filters instead of (64, 128, 256, 512)
# 500 epochs (!)
# 45000 images vs 50000
# LR (0.1, 0.01, 0.001) at (250, 375)
# Weight decay 1e-4, momentum 0.9, nesterov momentum

from __future__ import print_function

import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class StochasticPreActBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1, p_survive=0.5):
        super(StochasticPreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )
        
        self.p_survive = p_survive
        self.survival_count = 0
        self.drop_count = 0
    
    def _forward_train(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        if np.random.uniform(0, 1) < self.p_survive:
            self.survival_count += 1
            out = self.conv1(out)
            out = self.conv2(F.relu(self.bn2(out)))
            return out + shortcut
        else:
            self.drop_count += 1
            return shortcut
    
    def _forward_eval(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        return self.p_survive * out + shortcut # !! Would we do better if we did multiple passes w/o the adjustment
    
    def forward(self, x):
        if self.training:
            return self._forward_train(x)
        else:
            return self._forward_eval(x)
    
    def print_summary(self):
        print('p_survive=%f | drop_count=%d | survival_count=%d' % 
            (self.p_survive, self.drop_count, self.survival_count), file=sys.stderr)


class StochasticResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(StochasticResNet, self).__init__()
        self.in_planes = 64
        
        self.conv1  = conv3x3(3, 64)
        self.bn1    = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, p_survive=1.0)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, p_survive=0.5)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, p_survive=0.5)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, p_survive=0.5)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        
    def _make_layer(self, block, planes, num_blocks, stride, p_survive=0.5):
        strides = [stride] + [1] * (num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(in_planes=self.in_planes, planes=planes, stride=stride, p_survive=p_survive))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
        
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
    def eval(self):
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for child in layer.children():
                child.print_summary()
        
        return super(StochasticResNet, self).eval()
    
    def set_p_survive(self, ps):
        for p, layer in zip(ps, [self.layer1, self.layer2, self.layer3, self.layer4]):
            for child in layer.children():
                child.p_survive = p




def StochasticResNet18():
    return StochasticResNet(StochasticPreActBlock, [2,2,2,2])


def test():
    net = StochasticResNet18()
    y = net(Variable(torch.randn(1, 3, 32, 32)))
    print(y.size())

# test()
