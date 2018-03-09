#!/bin/bash

CUDA_VISIBLE_DEVICES=1 ./main.py --epochs 600 > ./results/resnet18-linear-600

# --
# Stochastic depth experiments

# Constant learning rates -- these really stink!
# CUDA_VISIBLE_DEVICES=1 ./main.py \
#     --lr-schedule constant \
#     --epochs 1000 > ./results/resnet18-constant-1000

# CUDA_VISIBLE_DEVICES=0 ./main.py \
#     --net stochastic_resnet18 \
#     --lr-schedule constant \
#     --epochs 1000 > ./results/stochastic-resnet18-constant-1000

# Linear decay
CUDA_VISIBLE_DEVICES=0 ./main.py \
    --net stochastic_resnet18 \
    --lr-schedule linear \
    --epochs 100 > ./results/stochastic-resnet18-linear-100

CUDA_VISIBLE_DEVICES=0 ./main.py \
    --net stochastic_resnet18 \
    --lr-schedule linear \
    --epochs 200 > ./results/stochastic-resnet18-linear-200

# Cyclical decay
CUDA_VISIBLE_DEVICES=1 ./main.py \
    --lr-schedule cyclical \
    --epochs 20 > ./results/resnet18-cyclical-20

CUDA_VISIBLE_DEVICES=0 ./main.py \
    --net stochastic_resnet18 \
    --lr-schedule cyclical \
    --epochs 20 > ./results/stochastic-resnet18-cyclical-20

CUDA_VISIBLE_DEVICES=1 ./main.py \
    --lr-schedule cyclical \
    --epochs 100 > ./results/resnet18-cyclical-100

CUDA_VISIBLE_DEVICES=0 ./main.py \
    --net stochastic_resnet18 \
    --lr-schedule cyclical \
    --epochs 100 > ./results/stochastic-resnet18-cyclical-100

CUDA_VISIBLE_DEVICES=0 ./main.py \
    --net stochastic_resnet18 \
    --lr-schedule cyclical \
    --reduce-p-survive \
    --epochs 100 > ./results/stochastic-resnet18-cyclical-100-red-1777

python utils/plot.py \
    ./results/resnet18-cyclical-100 \
    ./results/stochastic-resnet18-cyclical-100 \
    ./results/stochastic-resnet18-cyclical-100-red-1987 \
    ./results/stochastic-resnet18-cyclical-100-red-1777

# --
# Pipenet

CUDA_VISIBLE_DEVICES=1 ./main.py \
    --net pipenet \
    --lr-schedule cyclical \
    --epochs 20 > ./results/pipenet-cyclical-20

CUDA_VISIBLE_DEVICES=1 ./main.py \
    --net pipenet \
    --lr-schedule cyclical \
    --train-size 0.9 \
    --epochs 20 > ./results/pipenet-cyclical-20-0.900000

CUDA_VISIBLE_DEVICES=1 ./main.py \
    --net pipenet \
    --lr-schedule constant \
    --lr-init 0.01 \
    --epochs 20 > ./results/pipenet-constant-20-0.01

CUDA_VISIBLE_DEVICES=1 ./main.py \
    --net pipenet \
    --lr-schedule constant \
    --lr-init 0.1 \
    --epochs 50 > ./results/pipenet-constant-50-0.1

CUDA_VISIBLE_DEVICES=1 ./main.py \
    --net pipenet \
    --lr-schedule constant \
    --lr-init 0.01 \
    --train-size 0.9 \
    --epochs 50 > ./results/pipenet-constant-50-0.01-augment-0.900000

CUDA_VISIBLE_DEVICES=1 ./main.py --epochs 200 --lr-schedule linear --lr-init 0.1 > ./results/resnet18-linear-200.v2