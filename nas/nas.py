
"""
    nas.py
    
    Models for NAS
"""

import sys
import json
import numpy as np
from hashlib import md5
from datetime import datetime

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

class Identity(nn.Sequential):
    def forward(self, x):
        return x


class BNConv(nn.Sequential):
    def __init__(self, channels_in, channels_out, 
        kernel_size=3, padding=1, bias=False, stride=1,
        order=('bn', 'relu', 'op')):
        
        super(BNConv, self).__init__()
        
        self.add_module('bn', nn.BatchNorm2d(channels_in))
        self.add_module('relu', nn.ReLU())
        self.add_module('conv', nn.Conv2d(
            channels_in, 
            channels_out, 
            kernel_size=kernel_size, 
            padding=padding,
            bias=bias,
            stride=stride
        ))


class DoubleBNConv(nn.Sequential):
    def __init__(self, channels_in, channels_out, 
        kernel_size=3, padding=1, bias=False, stride=1,
        order=('bn', 'relu', 'op')):
        
        super(DoubleBNConv, self).__init__()
        
        self.add_module('bnconv.1', BNConv(
            channels_in, 
            channels_out, 
            kernel_size=kernel_size, 
            padding=padding,
            bias=bias,
            stride=stride
        ))
        self.add_module('bnconv.2', BNConv(
            channels_out, 
            channels_out, 
            kernel_size=kernel_size, 
            padding=padding,
            bias=bias,
            stride=1
        ))


ops = {
    "identity" : lambda c: Identity(),
    
    "bnconv_1" : lambda c: BNConv(c, c, kernel_size=1, padding=0),
    "bnconv_3" : lambda c: BNConv(c, c, kernel_size=3, padding=1),
    
    "double_bnconv_1" : lambda c: DoubleBNConv(c, c, kernel_size=1, padding=0),
    "double_bnconv_3" : lambda c: DoubleBNConv(c, c, kernel_size=3, padding=1),
    
    "avgpool_3" : lambda c: nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
    
    "maxpool_3" : lambda c: nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
    "maxpool_5" : lambda c: nn.MaxPool2d(kernel_size=5, stride=1, padding=2),
    "maxpool_7" : lambda c: nn.MaxPool2d(kernel_size=7, stride=1, padding=3),
    
    # sepconvs -- don't seem to be implemented to PyTorch
    # 1xn convs -- can skip for now I guess
}

red_ops = {
    "conv_1" : lambda c: nn.Conv2d(c, c * 2, kernel_size=1, padding=0, stride=2, bias=False),
    "conv_3" : lambda c: nn.Conv2d(c, c * 2, kernel_size=3, padding=1, stride=2, bias=False),
    
    "bnconv_1" : lambda c: BNConv(c, c * 2, kernel_size=1, padding=0, stride=2),
    "bnconv_3" : lambda c: BNConv(c, c * 2, kernel_size=3, padding=1, stride=2),
    
    "double_bnconv_1" : lambda c: DoubleBNConv(c, c * 2, kernel_size=1, padding=0, stride=2),
    "double_bnconv_3" : lambda c: DoubleBNConv(c, c * 2, kernel_size=3, padding=1, stride=2),
}

combs = {
    'add' : torch.add,
    'mul' : torch.mul,
}

class RBlock(nn.Module):
    """ Random convolutional block """
    def __init__(self, channels, ops, op_keys):
        super(RBlock, self).__init__()
        
        self.k1, self.k2, self.kc = op_keys
        
        self.op1 = ops[self.k1](channels)
        self.op2 = ops[self.k2](channels)
        self.comb = combs[self.kc]
    
    def forward(self, x):
        return self.comb(self.op1(x), self.op2(x))


class RNet(nn.Module):
    """ Random convolutional model """
    def __init__(self, op_keys, red_op_keys, block_sizes=(2, 1, 1, 1), init_channels=64, num_classes=10):
        super(RNet, self).__init__()
        
        # Can't both be pooling -- that's a silly model
        pool_1 = 'pool' in op_keys[0]
        pool_2 = 'pool' in op_keys[1]
        assert(not (pool_1 and pool_2))
        
        self.op_keys = op_keys
        self.red_op_keys = red_op_keys
        
        self.block_sizes = block_sizes
        self.init_channels = init_channels
        self.num_classes = num_classes
        
        self.conv0= nn.Conv2d(
            3, init_channels,
            kernel_size=3, 
            padding=1,
            bias=False,
            stride=1
        )
        self.bn0 = nn.BatchNorm2d(init_channels)
        self.relu0 = nn.ReLU()
        
        self.normal1 = self._make_normal(init_channels * 2 ** 0, block_sizes[0])
        self.reduction1 = self._make_reduction(init_channels * 2 ** 0)
        
        self.normal2 = self._make_normal(init_channels * 2 ** 1, block_sizes[1])
        self.reduction2 = self._make_reduction(init_channels * 2 ** 1)
        
        self.normal3 = self._make_normal(init_channels * 2 ** 2, block_sizes[2])
        self.reduction3 = self._make_reduction(init_channels * 2 ** 2)
        
        self.normal4 = self._make_normal(init_channels * 2 ** 3, block_sizes[3])
        
        self.linear = nn.Linear(init_channels * 2 ** 3, num_classes)
    
    def _make_normal(self, channels, block_size):
        tmp = []
        for _ in range(block_size):
            tmp.append(RBlock(channels, ops, self.op_keys))
            
        return nn.Sequential(*tmp)
    
    def _make_reduction(self, channels):
        return RBlock(channels, red_ops, self.red_op_keys)
    
    def forward(self, x):
        # Initial input
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu0(x)
        
        # Residual blocks
        x = self.normal1(x)
        x = self.reduction1(x)
        x = self.normal2(x)
        x = self.reduction2(x)
        x = self.normal3(x)
        x = self.reduction3(x)
        x = self.normal4(x)
        
        # Linear classifier
        out = F.avg_pool2d(x, x.size(-1))
        out = out.view(out.size(0), -1)
        return self.linear(out)
    
    def train_step(self, data, targets, optimizer):
        optimizer.zero_grad()
        outputs = self(data)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()
        return outputs, loss.data[0]


def sample_config():
    
    op_keys = sorted(np.random.choice(ops.keys(), 2)) + ['add']
    # !! Don't want double-pool blocks
    while ('pool' in op_keys[0]) and ('pool' in op_keys[1]):
        op_keys = sorted(np.random.choice(ops.keys(), 2)) + ['add']
    
    red_op_keys = sorted(np.random.choice(red_ops.keys(), 2)) + ['add']
    
    config = {
        'timestamp'   : datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
        'op_keys'     : op_keys,
        'red_op_keys' : red_op_keys,
    }
    
    config['model_name'] = md5(json.dumps(config)).hexdigest()
    return config

if __name__ == "__main__":
    op_keys = ('double_bnconv_3', 'identity', 'add')
    red_op_keys = ('double_bnconv_3', 'conv_1', 'add')
    net = RNet(op_keys, red_op_keys)
    print >> sys.stderr, net