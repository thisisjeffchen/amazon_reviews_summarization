#!/bin/bash

for b in 64 96 #128 160 192 224 256
do
    for d in 0 #1 2 3
    do
        echo "Batch: "
        echo $b
        echo "Device: "
        echo $d
        
        time CUDA_VISIBLE_DEVICES=$d python tf_cnn_benchmarks.py --num_gpus=1 --batch_size=$b --model=resnet50 --variable_update=parameter_server

        echo "Batch: "
        echo $b
        echo "Device: "
        echo $d
        echo "===USE FP16==="
        time CUDA_VISIBLE_DEVICES=$d python tf_cnn_benchmarks.py --num_gpus=1 --batch_size=$b --model=resnet50 --variable_update=parameter_server --use_fp16

    done
done
    

    

