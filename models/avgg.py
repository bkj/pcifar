
"""
    avgg.py
"""

from __future__ import print_function

import sys
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

class AVGG(nn.Module):
    def __init__(self, num_classes=10, input_dim=32):
        super(AVGG, self).__init__()
        self.features = self._make_layers([
            64, 64,
            128, 128,
            256, 256, 256, 'M',
            512, 512, 512, 'M',
            512, 512, 512, 'M',
            512, 'M',
            512, 'M'
        ])
        
        sz = self.features(Variable(torch.zeros(1, 3, input_dim, input_dim))).size()
        
        self.classifier = nn.Sequential(
            nn.Linear(np.prod(sz[1:]), 512),
            nn.Linear(512, num_classes),
        )
    
    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
        
    def _make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
        
                in_channels = v
        
        return nn.Sequential(*layers)

# --
# Resnet w/ attention

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )
            
    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class AResNet(nn.Module):
    def __init__(self, block=PreActBlock, num_blocks=[2,2,2,2], num_classes=10):
        super(AResNet, self).__init__()
        self.in_planes = 64
        
        self.conv1  = conv3x3(3,64)
        self.bn1    = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.fc1 = nn.Linear(512, 512)
        
        self.g1 = nn.Linear(512, 64, bias=False)
        self.a1 = nn.Linear(64, 1, bias=False)
        
        # self.g2 = nn.Linear(512, 128, bias=False)
        # self.a2 = nn.Linear(128, 1, bias=False)
        
        # self.g3 = nn.Linear(512, 256, bias=False)
        # self.a3 = nn.Linear(256, 1, bias=False)
        
        self.fc2 = nn.Linear(64, num_classes)
        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        
        return nn.Sequential(*layers)
    
    def forward(self, x, return_attention=False):
        # (Almost) normal pass
        prepped = F.relu(self.bn1(self.conv1(x)))
        
        block1 = self.layer1(prepped)
        block2 = self.layer2(block1)
        block3 = self.layer3(block2)
        block4 = self.layer4(block3)
        
        g = F.avg_pool2d(block4, 4)
        g = g.view(g.size(0), -1)
        g = self.fc1(g)
        
        # Attend to output of block2 -- careful, lots of room for error
        spatial_dim = 16
        n_channels = 128
        g2  = self.g2(g)
        g2  = g2.view(g2.size(0), g2.size(1), 1, 1)
        bg2 = block2 + g2
        bg2 = bg2.view(bg2.size(0), n_channels, spatial_dim ** 2)
        bg2 = bg2.transpose(1, 2)
        a2  = F.softmax(self.a2(bg2).squeeze())
        a2  = a2.view(a2.size(0), 1, spatial_dim, spatial_dim)
        b2  = (a2 * block2).sum(dim=-1).sum(dim=-1)
        
        # # Attend to output of block3 -- careful, lots of room for error
        # spatial_dim = 8
        # n_channels = 256
        # g3  = self.g3(g)
        # g3  = g3.view(g3.size(0), g3.size(1), 1, 1)
        # bg3 = block3 + g3
        # bg3 = bg3.view(bg3.size(0), n_channels, spatial_dim ** 2)
        # bg3 = bg3.transpose(1, 2)
        # a3  = F.softmax(self.a3(bg3).squeeze())
        # a3  = a3.view(a3.size(0), 1, spatial_dim, spatial_dim)
        # b3  = (a3 * block3).sum(dim=-1).sum(dim=-1)
        
        # out = self.fc2(torch.cat([b2, b3], dim=1))
        out = self.fc2(b2)
        if not return_attention:
            return out
        else:
            return a2, a3

# --


class AResNet2(nn.Module):
    def __init__(self, block=PreActBlock, num_blocks=[2,2,2,2], num_classes=10):
        super(AResNet2, self).__init__()
        self.in_planes = 64
        
        self.conv1  = conv3x3(3,64)
        self.bn1    = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.g1_0 = nn.Linear(512, 64, bias=False)
        self.g1_1 = nn.Linear(64, 64, bias=False)
        
        self.a1 = nn.Linear(64, 1, bias=False)
        
        self.fc = nn.Linear(64, num_classes)
        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        
        return nn.Sequential(*layers)
    
    def forward(self, x, return_attention=False):
        # (Almost) normal pass
        prepped = F.relu(self.bn1(self.conv1(x)))
        
        block1 = self.layer1(prepped)
        block2 = self.layer2(block1)
        block3 = self.layer3(block2)
        block4 = self.layer4(block3)
        
        g = F.max_pool2d(block4, 4)
        g = g.view(g.size(0), -1)
        
        spatial_dim = 32
        
        g = self.g1_1(F.tanh(self.g1_0(g)))
        g = g.view(g.size(0), g.size(1), 1, 1)
        
        a = (g * block1).sum(dim=1)
        a = a.view(a.size(0), spatial_dim ** 2)
        a = F.softmax(a)
        a = a.view(a.size(0), 1, spatial_dim, spatial_dim)
        
        out = (a * block1).sum(dim=-1).sum(dim=-1)
        out = self.fc(out)
        if not return_attention:
            return out
        else:
            return a

