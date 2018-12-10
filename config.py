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

parser.add_argument('--extractive_model', type=str, 
                    default='all',
                    help='the type of extractive model to use: kmeans, affinity, dbscan, pagerank')

parser.add_argument('--summary_length', type=int, 
                    default=5,
                    help='the number of sentences the extractive model must extract')

parser.add_argument('--products', type=str, 
                    default="three",
                    help='three or all, all will run through all with 50-100 reviews')


parser.add_argument('--debug', type=ast.literal_eval, 
                    default=True,
                    help='disable pdb breakpoints')

parser.add_argument('--abs_num_reviews', type=int, 
                    default=8,
                    help='the minimum number of reviews a product must have for ABS')

parser.add_argument('--learning_rate', type=float, 
                    default=0.005,
                    help='learning_rate')

parser.add_argument('--tie_in_out_embeddings', type=ast.literal_eval, 
                    default=False,
                    help='whether to initialize vocab projection layer with transpose of embedding matrix')

parser.add_argument('--use_pretrained_embeddings', type=ast.literal_eval, 
                    default=False,
                    help='whether to initialize from a pretrained embedding matrix')

parser.add_argument('--cold_start', type=ast.literal_eval, 
                    default=True,
                    help='whether to overwrite existing model_dir')

parser.add_argument('--prepare_embeddings', type=ast.literal_eval, 
                    default=False,
                    help='whether to run sentence encoders')

parser.add_argument('--embeddings_preprocessed', type=ast.literal_eval, 
                    default=True,
                    help='whether embeddings have been preprocessed and dont need to be extracted again for all reviews')


args = parser.parse_args()


DATA_PATH= os.environ.get('DATA_PATH', None) or './data/'
RESULTS_PATH = './results/'