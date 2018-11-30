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
import ipdb as pdb
from ipdb import slaunch_ipdb_on_exception

from config import DATA_PATH
MAX_SEQUENCE_LENGTH = 100
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 300
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
    with open('cache/tokenizer.pkl', 'wb') as fo:
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
    with open('cache/tokenizer.pkl', 'rb') as fi:
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


def abstractive_dataset_create(all_data_file= 'num_reviews.csv', num_reviews= 8):
    from sklearn.model_selection import train_test_split
    df= pd.read_csv(all_data_file, encoding='latin1')
    df1= df[df.num_reviews>=num_reviews].reset_index(drop=True)
    df_train, df_test= train_test_split(df1, test_size= .1, random_state=42)
    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)
    df_train.to_csv('abs_train_set_{}.csv'.format(num_reviews), index=False)
    df_test.to_csv('abs_test_set_{}.csv'.format(num_reviews), index=False)
    


def create_pretrained_embeddings(keras_tokenizer_fname= 'cache/tokenizer.pkl'):
    # first, build index mapping words in the embeddings set
    # to their embedding vector
    glove_embeddings_fname= os.environ.get('GLOVE_PATH', None)
    if glove_embeddings_fname is None:
        raise ValueError("Set GLOVE_PATH in bashrc")
    print('Indexing word vectors.')
    embeddings_index = {}
    with open(glove_embeddings_fname) as f:
        for i, line in enumerate(f):
            values = line.split()
            word = values[0]
            try:
                coefs = np.asarray(values[1:], dtype='float32')
            except ValueError:
                continue
            embeddings_index[word] = coefs
    print('Preparing embedding matrix.')
    with open(keras_tokenizer_fname, 'rb') as f:
        tokenizer= pickle.load(f)
    word_index = tokenizer.word_index
    # prepare embedding matrix
    num_words = min(MAX_NUM_WORDS, len(word_index)) + 1
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i > MAX_NUM_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    print(embedding_matrix.shape)
    np.save('cache/pretrained_embeddings.npy', embedding_matrix)


#create_pretrained_embeddings()
#if __name__ == "__main__":
#    build_tokenizer()
