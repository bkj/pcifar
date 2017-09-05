#!/bin/bash

mkdir -p results

# --
# ResNet18

CUDA_VISIBLE_DEVICES=1 ./main.py --epochs 10 --lr-schedule linear
CUDA_VISIBLE_DEVICES=1 ./main.py --epochs 25 --lr-schedule linear
CUDA_VISIBLE_DEVICES=1 ./main.py --epochs 50 --lr-schedule linear
CUDA_VISIBLE_DEVICES=1 ./main.py --epochs 100 --lr-schedule linear