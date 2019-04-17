#!/bin/bash

for d in 0 1 2 3
do
    echo "DEVICE:"
    echo $d
    time CUDA_VISIBLE_DEVICES=$d python sentiment_analysis/sentiment_train.py --epochs 40
done
