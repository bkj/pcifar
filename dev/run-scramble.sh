#!/bin/bash

mkdir -p results


CUDA_VISIBLE_DEVICES=1 ./main-scramble.py --epochs 25 --lr-init 0.2 --lr-smooth > ./results/scramble-0.2
