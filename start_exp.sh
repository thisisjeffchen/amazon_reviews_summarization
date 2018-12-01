#!/bin/bash
time CUDA_VISIBLE_DEVICES=0 python main_cluster.py --extractive_model=all --products=all --encoder_name=use > output_w_sentiment_use.txt
time CUDA_VISIBLE_DEVICES=0 python main_cluster.py --extractive_model=all --products=all --encoder_name=elmo > output_w_sentiment_elmo.txt
