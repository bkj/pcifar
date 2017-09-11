#!/bin/bash

# download.sh
# 
# Download + format datasets

# Download datasets, save to directories of images
python ./utils/download-datasets.py

# {train,test} -> {tv_train,tv_val,test}
python ./utils/validation-splits.py