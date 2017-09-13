#!/usr/bin/env python

"""
    enumerate-configs.py
"""

import sys
sys.path.append('..')

import os
import json
import argparse
import itertools
from hashlib import md5
from datetime import datetime
from nas import ops, red_ops, combs

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stdout', action='store_true')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    root = './results/enum-0/configs'
    
    ops_pairs = list(itertools.combinations_with_replacement(ops.keys(), 2))
    red_ops_pairs = list(itertools.combinations_with_replacement(red_ops.keys(), 2))
    
    ops_pairs = [p + ('add',) for p in ops_pairs] # hardcoded
    red_ops_pairs = [p + ('add',) for p in red_ops_pairs] # hardcoded
    
    for ok, rok in itertools.product(ops_pairs, red_ops_pairs):
        config = {'op_keys' : ok, 'red_op_keys' : rok}
        config['model_name'] = md5(json.dumps(config)).hexdigest()
        
        # Write to file
        if not args.stdout:
            json.dump(config, open(os.path.join(root, config['model_name']), 'w'))
        else:
            print json.dumps(config)