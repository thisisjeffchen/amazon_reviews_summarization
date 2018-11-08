#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 14:59:36 2018

@author: hanozbhathena
"""

import argparse
import ast
import os


parser= argparse.ArgumentParser()

parser.add_argument('--encoder_name', type=str, 
                    default='nnlm',
                    help='encoder type: can be nnlm, word2vec, use or elmo')

parser.add_argument('--trainable_embeddings', type=ast.literal_eval, 
                    default=False,
                    help='whether tf-hub encoder should be trainable')

parser.add_argument('--min_review_count', type=int, 
                    default=50,
                    help='the minimum number of reviews a product must have')

parser.add_argument('--all_review_countdf', type=str, 
                    default='num_reviews.csv',
                    help='the name of file containing the review counts per asin for all asins')

parser.add_argument('--filt_review_countdf', type=str, 
                    default='num_reviews_filt.csv',
                    help='the name of file containing the review counts per asin for filtered asins')

parser.add_argument('--train_chunksize', type=int, 
                    default=100,
                    help='the number of product asins queried at a time')

parser.add_argument('--train_batches', type=int, 
                    default=2,
                    help='the number of product groups taken each of chunksize')

parser.add_argument('--test_chunksize', type=int, 
                    default=100,
                    help='the number of product asins queried at a time')

parser.add_argument('--test_batches', type=int, 
                    default=1,
                    help='the number of product groups taken each of chunksize')

args = parser.parse_args()

DATA_PATH= os.environ.get('DATA_PATH') or './data'
