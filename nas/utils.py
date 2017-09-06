'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math

import torch.nn as nn
import torch.nn.init as init

_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.
        
    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1
    
    cur_time = time.time()
    last_time = cur_time
    tot_time = cur_time - begin_time
    
    L = []
    L.append('\tTime: %0.5f' % tot_time)
    if msg:
        L.append(' \t ' + msg)
        
    msg = ''.join(L)
    sys.stderr.write(msg)
    
    if current < total - 1:
        sys.stderr.write('\r')
    else:
        sys.stderr.write('\n')
    
    sys.stderr.flush()
