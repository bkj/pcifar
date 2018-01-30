#!/usr/bin/env python

import sys
import json
import numpy as np
from rsub import *
from matplotlib import pyplot as plt

def smart_json_loads(x):
    try:
        return json.loads(x)['test_acc']
    except:
        pass

all_data = []
for p in sys.argv[1:]:
    data = filter(None, map(smart_json_loads, open(p)))
    _ = plt.plot(data, alpha=0.75, label=p)

_ = plt.legend(loc='lower right')
_ = plt.axhline(0.9, c='grey', alpha=0.1)
_ = plt.axhline(0.95, c='grey', alpha=0.1)
show_plot()