#!/bin/bash

for i in $(seq 10000); do
    CUDA_VISIBLE_DEVICES=0 ./grid-point.py --run e100 --epochs 100
    CUDA_VISIBLE_DEVICES=0 ./grid-point.py --run e100 --lr-schedule cyclical --epochs 100
done
