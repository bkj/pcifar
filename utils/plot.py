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
    _ = plt.plot(data, alpha=0.25, label=p)
    all_data.append(data)

all_data = np.vstack(all_data)
np.save('.all_data', all_data)

# _ = plt.ylim(0.7, 1.0)
_ = plt.xlim(0, 5)
# _ = plt.legend(loc='lower right')
show_plot()