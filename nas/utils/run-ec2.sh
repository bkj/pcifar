#!/bin/bash

python enumerate-configs.py --stdout > configs

# First run
head -n 20 configs | python ec2.py --n-workers 4 --max-jobs 20

# Get finished ones
aws s3 ls s3://cfld-nas/results/ec2/CIFAR10/hists/ |\
    awk -F ' ' '{print $NF}' |\
    cut -d'-' -f1 > done

fgrep -v -f done configs | shuf > tmp
head -n 150 tmp | python ec2.py --n-workers 15 --max-jobs 150