#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 00:32:19 2018

@author: hanozbhathena
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import logging
logger = logging.getLogger("Training")
logger.setLevel(logging.INFO)
logging.basicConfig(format='%(levelname)s %(asctime)s : %(message)s', level=logging.INFO)
import ipdb as pdb
from ipdb import slaunch_ipdb_on_exception
from collections import defaultdict
import time
import dill as pickle

from config import DATA_PATH
from config import args
from text_encoders import NNLM, Word2Vec, USE, ELMO
from data_utils import SQLLiteBatchIterator, SQLLiteIndexer


def get_encoder():
    if args.encoder_name == 'nnlm':
        model= NNLM(trainable= args.trainable_embeddings)
    elif args.encoder_name == 'word2vec':
        model= Word2Vec(trainable= args.trainable_embeddings)
    elif args.encoder_name == 'use':
        model= USE(trainable= args.trainable_embeddings)
    elif args.encoder_name == 'elmo':
        model= ELMO(trainable= args.trainable_embeddings)
    else:
        raise TypeError("Invalid encoder name")
    return model


#TODO: make embeddings_np memory mapped for memory efficiency; will need to fix nrows and ncols
def encode_text_list(batch_iterator):
    """
    Takes a list of documents each is a string and converts to a 128-dim vector
    Example:
        docs= ["cat is on the mat", "dog is in the fog"]
        ret= (2, D) tensor where D is the dim of the encoder
            nnlm: 128
            word2vec: 500
            use: 512
            elmo: 1024
    """
    encoder= get_encoder()
    embeddings_list= []
    for batch_ind, review_batch in enumerate(batch_iterator):
        for i, text_list in enumerate(review_batch):
            batch_embeddings= encoder(text_list)
            embeddings_list.append(batch_embeddings)
            if i>0 and i%100==0:
                print(batch_ind, i)
    embeddings_np= np.concatenate(embeddings_list, axis= 0)
    return embeddings_np


def train_features(asin_df_file,
                   train_chunksize= args.train_chunksize, 
                   train_batches= args.train_batches, 
                   train_offset= 0, 
                   path= DATA_PATH):
    train_batcher= SQLLiteBatchIterator(asin_df_file, path, asin_chunksize= train_chunksize,
                                       num_chunks= train_batches, offset= train_offset)
    train_embeddings= encode_text_list(train_batcher)
    return train_embeddings


def test_features(asin_df_file,
                  test_chunksize= args.test_chunksize, 
                  test_batches= args.test_batches, 
                  test_offset= 0, 
                  path= DATA_PATH):
    test_batcher= SQLLiteBatchIterator(asin_df_file, path, asin_chunksize= test_chunksize,
                                       num_chunks= test_batches, offset= test_offset)
    test_embeddings= encode_text_list(test_batcher)
    return test_embeddings


def main(params):
    all_df= pd.read_csv(params['all_review_countdf'], encoding='latin1')
    df2use= all_df[all_df.num_reviews > params['min_review_count']]
    df2use= df2use.sample(frac=1).reset_index(drop=True)
    df2use_train= df2use.iloc[:int(len(df2use)*.5), :]
    df2use_test= df2use.iloc[int(len(df2use)*.5):, :]
    
    df2use_train.to_csv('df2use_train.csv', index=False)
    df2use_test.to_csv('df2use_test.csv', index=False)
    train_embeddings= train_features(asin_df_file= 'df2use_train.csv')
    with open(os.path.join(DATA_PATH, 'train_embeddings.pkl'), 'wb') as fo:
        pickle.dump(train_embeddings, fo, pickle.HIGHEST_PROTOCOL)
#    test_embeddings= test_features(asin_df_file= 'df2use_test.csv')
#    with open(os.path.join(DATA_PATH, 'test_embeddings.pkl'), 'wb') as fo:
#        pickle.dump(test_embeddings, fo, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    with slaunch_ipdb_on_exception():
        main(vars(args))

