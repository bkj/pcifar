#!/usr/bin/env python

"""
    enumerate-configs.py
"""

import os
import json
import itertools
from hashlib import md5
from datetime import datetime
from nas import ops, red_ops, combs

# --
# Enumerate all architectures in the currect framework

root = './results/enum-0/configs'

ops_pairs = list(itertools.combinations_with_replacement(ops.keys(), 2))
red_ops_pairs = list(itertools.combinations_with_replacement(red_ops.keys(), 2))

ops_pairs = [p + ('add',) for p in ops_pairs]
red_ops_pairs = [p + ('add',) for p in red_ops_pairs]

for ok, rok in itertools.product(ops_pairs, red_ops_pairs):
    config = {
        'op_keys'     : ok,
        'red_op_keys' : rok,
    }
    
    config['model_name'] = md5(json.dumps(config)).hexdigest()
    
    # Write to file
    json.dump(config, open(os.path.join(root, config['model_name']), 'w'))