#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 17:43:30 2018

@author: hanozbhathena
"""

# =============================================================================
# Model data utils
# =============================================================================
import os
import sys
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import dill as pickle
import sqlite3
import ast
import random
import tensorflow as tf

DATA_PATH= os.environ['DATA_PATH']
MAX_SEQUENCE_LENGTH = 100
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100
OOV_TOKEN= '<OOV>'
BATCH_SIZE= 32
NUM_EPOCHS= 10
NUM_REVIEWS_K= 16


def db_cur_gen(cur):
    for i, c in enumerate(cur):
        if i > 0 and i % 10000 == 0:
            print(i)
            break
        for text in ast.literal_eval(c[1]):
            yield text


def build_tokenizer(data_path= DATA_PATH, db_name= "reviews.s3db", asins2use_file= "df2use_train.csv"):
    tokenizer= Tokenizer(num_words=MAX_NUM_WORDS, lower= True, oov_token= OOV_TOKEN)
    review_iterator= TFReviewIterator(data_path, db_name, asins2use_file)
    tokenizer.fit_on_texts(db_cur_gen(review_iterator))
    with open('tokenizer.pkl', 'wb') as fo:
        pickle.dump(tokenizer, fo, pickle.HIGHEST_PROTOCOL)
    print("Saved Keras tokenizer")


class TFReviewIterator(object):
    def __init__(self, data_path= DATA_PATH, 
                 db_name= "reviews.s3db", 
                 asins2use_file= "df2use_train.csv"):
        db_file= os.path.join(data_path, db_name)
        asin_df= pd.read_csv(asins2use_file, encoding= 'latin1')
        self.asin_list= asin_df['asin'].tolist()
        self.query= """
             SELECT asin, reviewText from reviews_dict 
             where asin in ({num_qs})
             """.format(num_qs=','.join('?'*len(self.asin_list)))
        self.conn= sqlite3.connect(db_file, check_same_thread=False)
        self.cur= self.conn.cursor()
    
    def __iter__(self):
        self.cur.execute(self.query, self.asin_list)
        for tup in self.cur:
            yield tup
    
    def __del__(self):
        print("Closing SQLLite connection")
        self.conn.close()



def train_input_fn(data_path= DATA_PATH, db_name= "reviews.s3db", asins2use_file= "df2use_train.csv"):
    with open('tokenizer.pkl', 'rb') as fi:
        tokenizer= pickle.load(fi)
    review_iterator= TFReviewIterator(data_path, db_name, asins2use_file)
    
    # like https://github.com/keras-team/keras/blob/master/examples/pretrained_word_embeddings.py
    def tf_data_gen(K= NUM_REVIEWS_K, maxlen= MAX_SEQUENCE_LENGTH):
        for i, (asin, review_str_list) in enumerate(review_iterator):
            review_list= ast.literal_eval(review_str_list)
            random.shuffle(review_list)
            review_list= review_list[:K]
            asin_list= [asin]*len(review_list)
            id_list= tokenizer.texts_to_sequences(review_list)
            data_batch= pad_sequences(id_list, maxlen=maxlen, dtype= np.int32, 
                                      padding='post', truncating='post')
            text_list= tokenizer.sequences_to_texts([ids[:maxlen] for ids in id_list])
            real_lens= np.array([len(text.split()) for text in text_list]).astype(np.int32)
            ret_dict= {'asin_list': asin_list,
                       'text_list': text_list,
                       'data_batch': data_batch,
                       'real_lens': real_lens,
                       }
#            yield asin_list, text_list, data_batch, real_lens
            yield ret_dict
    
#    ds= tf.data.Dataset.from_generator(
#        tf_data_gen, (tf.string, tf.string, tf.int32, tf.int32), 
#        (tf.TensorShape([None]), tf.TensorShape([None]), 
#         tf.TensorShape([None, MAX_SEQUENCE_LENGTH]), tf.TensorShape([None]))
#        )
    
    ds= tf.data.Dataset.from_generator(
    tf_data_gen, {'asin_list': tf.string, 'text_list': tf.string, 'data_batch': tf.int32, 'real_lens': tf.int32}, 
    {'asin_list': tf.TensorShape([None]), 'text_list': tf.TensorShape([None]), 
     'data_batch': tf.TensorShape([None, MAX_SEQUENCE_LENGTH]), 'real_lens': tf.TensorShape([None])}
    )
    
    dataset= ds.shuffle(1000).repeat(NUM_EPOCHS)
#    dataset= dataset.batch(BATCH_SIZE)
    
    return dataset


def test():
    dataset= train_input_fn()
    value = dataset.make_one_shot_iterator().get_next()
    sess= tf.Session()
    tup= sess.run(value)
    return value

