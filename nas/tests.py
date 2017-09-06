import torch
from nas import *

# Test that all ops preserve size
def test_ops(ops, orig_size=(16, 3, 10, 10)):
    x = Variable(torch.rand(orig_size))
    for k, op in ops.iteritems():
        f = ops[k](orig_size[1])
        assert f(x).size() == orig_size

test_ops(ops)

