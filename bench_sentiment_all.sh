#!/bin/bash

for d in 0 1 2 3
do
    echo "DEVICE:"
    echo $d
    CUDA_VISIBLE_DEVICES=$d time python sentiment_analysis/sentiment_train.py --epochs 10
done
