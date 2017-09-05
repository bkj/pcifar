#!/bin/bash

mkdir -p results

# --
# ResNet18

CUDA_VISIBLE_DEVICES=1 ./main.py --epochs 10 > ./results/resnet18-linear-10

# !! Retrying w/ learning rate that changes every batch
CUDA_VISIBLE_DEVICES=1 ./main.py --epochs 10 --lr-smooth > ./results/resnet18-linear-10-smooth

CUDA_VISIBLE_DEVICES=1 ./main.py --epochs 25 > ./results/resnet18-linear-25
CUDA_VISIBLE_DEVICES=1 ./main.py --epochs 50 > ./results/resnet18-linear-50
CUDA_VISIBLE_DEVICES=1 ./main.py --epochs 75 > ./results/resnet18-linear-75
CUDA_VISIBLE_DEVICES=1 ./main.py --epochs 100 > ./results/resnet18-linear-100
CUDA_VISIBLE_DEVICES=1 ./main.py --epochs 150 > ./results/resnet18-linear-150
CUDA_VISIBLE_DEVICES=1 ./main.py --epochs 200 > ./results/resnet18-linear-200
CUDA_VISIBLE_DEVICES=1 ./main.py --epochs 250 > ./results/resnet18-linear-250
CUDA_VISIBLE_DEVICES=1 ./main.py --epochs 300 > ./results/resnet18-linear-300
CUDA_VISIBLE_DEVICES=1 ./main.py --epochs 350 > ./results/resnet18-linear-350

# --
# Cyclical learning rates

CUDA_VISIBLE_DEVICES=1 ./main.py --epochs 25 --lr-schedule cyclical --lr-smooth --lr-init 0.1 > ./results/resnet18-linear-25-cyc-0.1
CUDA_VISIBLE_DEVICES=1 ./main.py --epochs 50 --lr-schedule cyclical --lr-smooth --lr-init 0.1 > ./results/resnet18-linear-50-cyc-0.1

CUDA_VISIBLE_DEVICES=1 ./main.py --epochs 25 --lr-schedule cyclical --lr-smooth --lr-init 0.2 > ./results/resnet18-linear-25-cyc-0.2
CUDA_VISIBLE_DEVICES=1 ./main.py --epochs 50 --lr-schedule cyclical --lr-smooth --lr-init 0.2 > ./results/resnet18-linear-50-cyc-0.2

# --
# Cyclical learning rates (VGG)

NET=googlenet

CUDA_VISIBLE_DEVICES=1 ./main.py --net $NET --epochs 25 --lr-smooth > ./results/$NET-linear-25
CUDA_VISIBLE_DEVICES=1 ./main.py --net $NET --epochs 25 --lr-schedule cyclical --lr-smooth > ./results/$NET-linear-25-cyc

CUDA_VISIBLE_DEVICES=1 ./main.py --net $NET --epochs 50 --lr-smooth > ./results/$NET-linear-50
CUDA_VISIBLE_DEVICES=1 ./main.py --net $NET --epochs 50 --lr-schedule cyclical --lr-smooth > ./results/$NET-linear-50-cyc

CUDA_VISIBLE_DEVICES=1 ./main.py --net $NET --epochs 100 --lr-smooth > ./results/$NET-linear-100
CUDA_VISIBLE_DEVICES=1 ./main.py --net $NET --epochs 100 --lr-schedule cyclical --lr-smooth > ./results/$NET-linear-100-cyc
