#!/bin/bash

for CONFIG in $(find results/enum-0/configs/ -type f); do
    echo $CONFIG
    CUDA_VISIBLE_DEVICES=0 python grid-point.py \
        --run enum-0 \
        --config $CONFIG \
        --epochs 2 \
        --lr-schedule constant \
        --train-history
done


# ... create results/enum-0/good-e2 -- names of configs that perform well at e2 ...

find ./results/enum-0/hists -size +1k | xargs -I {} basename {} | shuf > tmp
for CONFIG in $(cat tmp); do
    echo $CONFIG
    CUDA_VISIBLE_DEVICES=0 python grid-point.py \
        --run enum-0 \
        --config ./results/enum-0/configs/$CONFIG \
        --hot-start \
        --epochs 10 \
        --lr-schedule constant \
        --train-history
done