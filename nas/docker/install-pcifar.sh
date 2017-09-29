#!/bin/bash

mkdir -p ~/projects
cd ~/projects

git clone https://github.com/bkj/pcifar -b nas
cd pcifar/nas

# Download + format datasets
./utils/download.sh
