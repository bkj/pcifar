#!/bin/bash

for i in $(seq 100); do ./grid-point.py --run 1; done
for i in $(seq 100); do ./grid-point.py --run 1 --lr-schedule cyclical ; done

for i in $(seq 1000); do ./grid-point.py --run 1; done