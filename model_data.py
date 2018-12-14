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
from collections import defaultdict
from nltk.tokenize import sent_tokenize

from config import DATA_PATH, args
MAX_SEQUENCE_LENGTH = 100
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 300
OOV_TOKEN= '<OOV>'
BATCH_SIZE= 1 #number of products per batch
NUM_EPOCHS= 5
NUM_REVIEWS_K= args.abs_num_reviews #number of reviews per product
EOS= ' <EOS> '

def db_cur_gen(cur):
    for i, c in enumerate(cur):
        if i > 0 and i % 10000 == 0:
            print(i)
        #     break
        for text in ast.literal_eval(c[1]):
            sents= sent_tokenize(text)
            text= EOS.join(sents) + EOS
            yield text


def build_tokenizer(data_path= DATA_PATH, db_name= "reviews.s3db", asins2use_file= "abs_train_set_8.csv"):
    # pdb.set_trace()
    tokenizer= Tokenizer(num_words=MAX_NUM_WORDS, lower= True, oov_token= OOV_TOKEN)
    review_iterator= TFReviewIterator(data_path, db_name, asins2use_file)
    tokenizer.fit_on_texts(db_cur_gen(review_iterator))
    with open('cache/tokenizer.pkl', 'wb') as fo:
        pickle.dump(tokenizer, fo, pickle.HIGHEST_PROTOCOL)
    print("Saved Keras tokenizer")


class TFReviewIterator(object):
    def __init__(self, data_path= DATA_PATH, 
                 db_name= "reviews.s3db", 
                 asins2use_file= "abs_train_set_8.csv"):
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



# def train_input_fn_gen(data_path= DATA_PATH, db_name= "reviews.s3db", asins2use_file= "abs_train_set_8.csv"):
#     with open('cache/tokenizer.pkl', 'rb') as fi:
#         tokenizer= pickle.load(fi)
#     review_iterator= TFReviewIterator(data_path, db_name, asins2use_file)
    
#     # like https://github.com/keras-team/keras/blob/master/examples/pretrained_word_embeddings.py
#     def tf_data_gen(K= NUM_REVIEWS_K, maxlen= MAX_SEQUENCE_LENGTH):
#         for i, (asin, review_str_list) in enumerate(review_iterator):
#             review_list= ast.literal_eval(review_str_list)
#             random.shuffle(review_list)
#             review_list= review_list[:K]
#             asin_list= [asin]*len(review_list)
#             id_list= tokenizer.texts_to_sequences(review_list)
#             data_batch= pad_sequences(id_list, maxlen=maxlen, dtype= np.int32, 
#                                       padding='post', truncating='post')
#             text_list= tokenizer.sequences_to_texts([ids[:maxlen] for ids in id_list])
#             real_lens= np.array([len(text.split()) for text in text_list]).astype(np.int32)
#             ret_dict= {'asin_list': asin_list,
#                        'text_list': text_list,
#                        'data_batch': data_batch,
#                        'real_lens': real_lens,
#                        }
#             yield ret_dict
    
#     ds= tf.data.Dataset.from_generator(
#     tf_data_gen, {'asin_list': tf.string, 'text_list': tf.string, 'data_batch': tf.int32, 'real_lens': tf.int32}, 
#     {'asin_list': tf.TensorShape([None]), 'text_list': tf.TensorShape([None]), 
#      'data_batch': tf.TensorShape([None, MAX_SEQUENCE_LENGTH]), 'real_lens': tf.TensorShape([None])}
#     )
    
#     dataset= ds.shuffle(1000).repeat(NUM_EPOCHS)
#     dataset= dataset.batch(BATCH_SIZE)
#     return dataset


def append_eos(review_list):
    ret_list= []
    for text in review_list:
        sents= sent_tokenize(text)
        text= EOS.join(sents) + EOS
        ret_list.append(text)
    return ret_list


def prepare_df(data_path= DATA_PATH, db_name= "reviews.s3db", asins2use_file= "abs_train_set_8.csv"):
    with open('cache/tokenizer.pkl', 'rb') as fi:
        tokenizer= pickle.load(fi)
    review_iterator= TFReviewIterator(data_path, db_name, asins2use_file)
    
    def tf_data_df(K= NUM_REVIEWS_K, maxlen= MAX_SEQUENCE_LENGTH):
        # pdb.set_trace()
        ddict= defaultdict(list)
        data_batch_list= []
        for i, (asin, review_str_list) in enumerate(review_iterator):
            review_list= ast.literal_eval(review_str_list)
            # review_list= append_eos(review_list) #TODO: causing memory error in tensorflow: NO IDEA WHY CHECK!!!
            random.shuffle(review_list)
            review_list= review_list[:K]
            asin_list= [asin]*len(review_list)
            asin_num_list= [i]*len(review_list)
            id_list= tokenizer.texts_to_sequences(review_list)
            data_batch= pad_sequences(id_list, maxlen=maxlen, dtype= np.int32, 
                                      padding='post', truncating='post')
            text_list= tokenizer.sequences_to_texts([ids[:maxlen] for ids in id_list])
            real_lens= np.array([len(text.split()) for text in text_list]).astype(np.int32)
            ddict['asin_list'].extend(asin_list)
            ddict['asin_num_list'].extend(asin_num_list)
            ddict['text_list'].extend(text_list)
            ddict['real_lens'].extend(real_lens)

            data_batch_list.append(data_batch)
            # if i > 10000:
            #     break
        ret_df= pd.DataFrame(ddict)
        word_ids= np.vstack(data_batch_list)
        return ret_df, word_ids
    
    features_df, word_ids= tf_data_df()
    print("Features Dataframe shape: {}".format(features_df.shape))
    print("Word IDs data batch Dataframe shape: {}".format(word_ids.shape))
    return features_df, word_ids


def train_input_fn(features_df, word_ids):       
    pdb.set_trace()
    features= dict(features_df)
    features['data_batch']= word_ids
    # pdb.set_trace()
    ds = tf.data.Dataset.from_tensor_slices(features)
    dataset= ds.prefetch(NUM_REVIEWS_K*BATCH_SIZE).repeat(NUM_EPOCHS).batch(NUM_REVIEWS_K*BATCH_SIZE)
    # dataset= ds.repeat(NUM_EPOCHS).batch(NUM_REVIEWS_K*BATCH_SIZE)
    return dataset


def test():
    dataset= train_input_fn()
    value = dataset.make_one_shot_iterator().get_next()
    sess= tf.Session()
    tup= sess.run(value)
    return value


def abstractive_dataset_create(all_data_file= 'num_reviews.csv', num_reviews= args.abs_num_reviews):
    from sklearn.model_selection import train_test_split
    df= pd.read_csv(all_data_file, encoding='latin1')
    df1= df[df.num_reviews>=num_reviews].reset_index(drop=True)
    df_train, df_test= train_test_split(df1, test_size= .1, random_state=42)
    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)
    train_filename, test_filename= 'abs_train_set_{}.csv'.format(num_reviews), 'abs_test_set_{}.csv'.format(num_reviews)
    df_train.to_csv(train_filename, index=False)
    df_test.to_csv(test_filename, index=False)
    return train_filename, test_filename


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



if __name__ == "__main__":
    with slaunch_ipdb_on_exception():
        # train_filename, test_filename= abstractive_dataset_create()
        train_filename= 'abs_train_set_8.csv'
        build_tokenizer(asins2use_file= train_filename)
        # create_pretrained_embeddings()
