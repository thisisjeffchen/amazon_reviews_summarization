#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 14:47:39 2018

@author: hanozbhathena
"""
import gc
import sys
import os
import pandas as pd
import numpy as np
import logging
logger = logging.getLogger("Training")
logger.setLevel(logging.INFO)
logging.basicConfig(format='%(levelname)s %(asctime)s : %(message)s', level=logging.INFO)
import ipdb as pdb
from ipdb import slaunch_ipdb_on_exception
from collections import defaultdict, OrderedDict
import time
import dill as pickle
import json
from nltk.tokenize import sent_tokenize
import sqlite3

import config
from config import args
from main_encode import get_encoder
from data_utils import SQLLiteBatchIterator, SQLLiteIndexer, SQLLiteEmbeddingsIndexer, SQLLiteAsinAttrIterator
from extractive_summ_modules import get_ex_summarizer, MyRouge, PreprocessEncoder
from sentiment_analysis.sentiment_model import CNN
from keras.models import load_model
from sentiment_analysis.data.util import pad_sentence
import tensorflow as tf

SENTENCE_LENGTH = 600


def write_json(kwargs, summary_dict, model):
    file_id = model if kwargs["products"] == "three" else model + "-1000"
    with open(config.RESULTS_PATH + 'summary_dict_proposal_{}_{}.json'.format(file_id, args.encoder_name), 'w') as fo:
        json.dump(summary_dict, fo, ensure_ascii=False, indent=2)

def load_sentiment_evaluator ():
    print ("Loading Models...")
    with open('cache/tokenizer_100000_sentiment.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    #TODO: ugly with hard coded values, can we just load the entire model?
    model = CNN(512, 6000, SENTENCE_LENGTH, 512, 3,0.95)
    model.load_weights("cache/amazon_100000_model.h5")
    #model = load_model('cache/amazon_100000_model.h5', custom_objects = {'CNN':CNN})

    print ("Models loaded...")
    return model, tokenizer

def evaluate_sentiment (reviews_short, summary, sent_model, sent_tokenizer):
    summary_toks = sent_tokenizer.texts_to_sequences([" ".join (summary)])
    reviews_short_toks = sent_tokenizer.texts_to_sequences (reviews_short)


    summary_processed = np.array([np.array(pad_sentence(summary_toks[0], SENTENCE_LENGTH))])
    reviews_short_processed = []
    for review in reviews_short_toks:
        review = pad_sentence (review, SENTENCE_LENGTH)
        reviews_short_processed.append (review)
    reviews_short_processed = np.array (reviews_short_processed)

    summary_hat = sent_model.predict (summary_processed)[0]
    reviews_short_hat = sent_model.predict (reviews_short_processed)

    reviews_short_max_idx = np.argmax (reviews_short_hat, axis = 1)
    total_score = 0
    for idx in reviews_short_max_idx:
        if idx == 1:
            total_score += 1
        elif idx == 2:
            total_score -= 1

    reviews_short_avg = round (total_score / len (reviews_short_hat))
    summary_score = np.argmax(summary_hat) 
    if summary_score == 2:
        summary_score = -1 #minus one because we go from 0, 1, -1 in labels

    return reviews_short_avg, summary_score, int (reviews_short_avg == summary_score)


def insert_emb(cur, asin, product_embs, product_sentences, sentence_parent):
    cur.execute("insert into reviews_embeddings (asin, product_embs, product_sentences, sentence_parent) values (?, ?, ?, ?)", 
                (str(asin), str(product_embs), str(product_sentences), str(sentence_parent)))


def insert_emb_many(cur, values_to_insert):
    cur.executemany("insert into reviews_embeddings (asin, product_embs, product_sentences, sentence_parent) values (?, ?, ?, ?)", values_to_insert)


def insert_emb_csv(cur, values_to_insert, write):
    pdb.set_trace()
    temp_df= pd.DataFram(values_to_insert)
    if write == True:
        temp_df.to_csv('test_db.csv', index= False)
    cur.executemany("insert into reviews_embeddings (asin, product_embs, product_sentences, sentence_parent) values (?, ?, ?, ?)", values_to_insert)


def input_data_fn(asin_list):
    reviews_iterator= SQLLiteAsinAttrIterator(asin_list)
    def tf_data_gen():
        for row_tup in reviews_iterator:
            asin, product_reviews= row_tup
            product_sentences, sentence_parent= [], []
            for idx, review in enumerate(product_reviews):
                for sent in sent_tokenize(review):
                    product_sentences.append(sent)
                    sentence_parent.append(idx)
            ret_dict= {'asin': asin,
                       'product_sentences': product_sentences,
                       'sentence_parent': sentence_parent,
                      }
            yield ret_dict
    
    ds= tf.data.Dataset.from_generator(
    tf_data_gen, {'asin': tf.string, 'product_sentences': tf.string, 'sentence_parent': tf.int32}, 
    {'asin': tf.TensorShape([None]), 'product_sentences': tf.TensorShape([None]), 'sentence_parent': tf.TensorShape([None])}
    )
    
    dataset= ds.repeat(1)
#    dataset= dataset.batch(BATCH_SIZE)
    return dataset



def hub_encoder(features, labels, mode, params):
    pdb.set_trace()
    hub_model= get_encoder()
    product_embs= hub_model(features['product_sentences'])
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'asin': features['asin'],
            'product_sentences': features['product_sentences'],
            'sentence_parent': features['sentence_parent'],
            'product_embs': product_embs,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    assert False


def get_df_from_db(kwargs):
    df= pd.read_csv('df2use_train.csv', encoding='latin1')
    df_filt= df[df.num_reviews<=100].reset_index(drop=True)
    if kwargs["products"] == "three":
        asin_list= ['B00008OE43', 'B0007OWASE', 'B000EI0EB8']
    elif kwargs["products"] == "all":
        asin_list= df_filt.asin.tolist()[:]
        # Cap at 1000 products
        asin_list = asin_list[0:1000]
    else:
        raise Exception ("Product group not recognized")
    reviews_iterator= SQLLiteAsinAttrIterator(asin_list)
    ddict= defaultdict(list)
    for i, row_tup in enumerate(reviews_iterator):
        asin, product_reviews= row_tup
        ddict['asin'].append(asin)
        ddict['product_reviews'].append(product_reviews)
    ret_df= pd.DataFrame(ddict)
    return ret_df


def encode_reviews(reviews_df, encoder, kwargs):
    preprocess_module= PreprocessEncoder(kwargs['summary_length'], embeddings_preprocessed=False)
    ddict= defaultdict(list)
    for i, row_tup in enumerate(reviews_df.itertuples()):
        asin, product_reviews= row_tup.asin, row_tup.product_reviews
        product_sentences, product_embs, sentence_parent= preprocess_module(asin, product_reviews, encoder)
        ddict['asin'].append(asin)
        ddict['product_sentences'].append(product_sentences)
        ddict['product_embs'].append(product_embs)
        ddict['sentence_parent'].append(sentence_parent)
    return ddict


def feed_embeddings_to_db(kwargs, ddict):
    db_file= os.path.join(config.DATA_PATH, "embedding_db-{}.s3db".format(args.encoder_name))
    conn= sqlite3.connect(db_file)
    cur= conn.cursor()
    cur.execute("drop table if exists reviews_embeddings;")
    cur.execute("CREATE TABLE IF NOT EXISTS reviews_embeddings ("
      "asin VARCHAR(255) PRIMARY KEY NOT NULL, "
      "product_embs VARCHAR(255),"
      "product_sentences VARCHAR(255),"
      "sentence_parent VARCHAR(255))")
    
    
def main_preprocess_embeddings(kwargs):
    pdb.set_trace()
    db_file= os.path.join(config.DATA_PATH, "embedding_db-{}.s3db".format(args.encoder_name))
    conn= sqlite3.connect(db_file)
    cur= conn.cursor()
    cur.execute("drop table if exists reviews_embeddings;")
    cur.execute("CREATE TABLE IF NOT EXISTS reviews_embeddings ("
      "asin VARCHAR(255) PRIMARY KEY NOT NULL, "
      "product_embs VARCHAR(255),"
      "product_sentences VARCHAR(255),"
      "sentence_parent VARCHAR(255))")

    preprocess_module= PreprocessEncoder(kwargs['summary_length'], embeddings_preprocessed=False)
    encoder= get_encoder()
    df= pd.read_csv('df2use_train.csv', encoding='latin1')
    df_filt= df[df.num_reviews<=100].reset_index(drop=True)
    if kwargs["products"] == "three":
        asin_list= ['B00008OE43', 'B0007OWASE', 'B000EI0EB8']
    elif kwargs["products"] == "all":
        asin_list= df_filt.asin.tolist()[:]
        # Cap at 1000 products
        asin_list = asin_list[0:1000]
    else:
        raise Exception ("Product group not recognized")
    # reviews_indexer= SQLLiteIndexer(config.DATA_PATH)
    try:
        reviews_iterator= SQLLiteAsinAttrIterator(asin_list)
        values_to_insert= []
        ddict= defaultdict(list)
        write= True
        for i, row_tup in enumerate(reviews_iterator):
            asin, product_reviews= row_tup
            product_sentences, product_embs, sentence_parent= preprocess_module(asin, product_reviews, encoder)
            # product_sentences, product_embs, sentence_parent= product_reviews, np.random.rand(len(product_reviews)*100, 500), np.random.randint(0,1000, 100*len(product_reviews))
            # insert_emb(cur, asin, product_embs.tolist(), product_sentences, sentence_parent)
            values_to_insert.append((str(asin), str(product_embs.tolist()), str(product_sentences), str(sentence_parent)))
            
            # ddict['asin'].append(asin)
            # ddict['product_embs'].append(product_embs.tolist())
            # ddict['product_sentences'].append(product_sentences)
            # ddict['sentence_parent'].append(sentence_parent)
            logging.info(i)
            gc.collect()

            if i > 0 and i % 50 == 0:
                insert_emb_many(cur, values_to_insert)
                # insert_emb_csv(cur, values_to_insert, write)
                write= False
                ddict= defaultdict(list)
                logging.info("Inserted {} total products to embeddings_db".format(i+1))
                values_to_insert= []

            if i > 0 and i % 500 == 0:
                conn.commit()
                logging.info("Commited {} total products to embeddings_db".format(i+1))
                cur.execute('select count(*) from reviews_embeddings')
                logging.info(cur.fetchone())
                gc.collect()
            sys.stdout.flush()
        pdb.set_trace()
        insert_emb_many(cur, values_to_insert)
        conn.commit()
        logging.info("Finished {} total products to embeddings_db".format(i+1))
        cur.execute('select count(*) from reviews_embeddings')
        logging.info(cur.fetchone())
        pdb.set_trace()
        test_indexer= SQLLiteEmbeddingsIndexer(args.encoder_name)
        ddict= test_indexer[asin]
        assert ddict['asin'] == asin
        assert ddict['product_sentences'] == product_sentences
        assert ddict['sentence_parent'] == sentence_parent
        np.testing.assert_equal(ddict['product_embs'], product_embs)
    except KeyboardInterrupt:
        logging.info("Keyboard interrup, commiting remaining")
        conn.commit()
    except Exception as e:
        logging.info("Error type: {}".format(type(e).__name__))
        conn.rollback()
    finally:
        cur.close()
        conn.close()


def main(kwargs):
    pdb.set_trace()
    if kwargs['extractive_model'] == "all":
        models = ["kmeans", "affinity", "dbscan", "pagerank"]
    elif kwargs['extractive_model'] == "three":
        models = ["affinity", "kmeans", "pagerank"]
    else:
        models = [kwargs['extractive_model']]

    sent_model, sent_tokenizer = load_sentiment_evaluator ()

    products_skipped= 0
    for model in models:
        summarization_module= get_ex_summarizer(model_type= model,
                                                summary_length= kwargs['summary_length'],
                                                embeddings_preprocessed= kwargs['embeddings_preprocessed'])
        rouge_module= MyRouge()
        encoder= get_encoder()
        reviews_indexer= SQLLiteIndexer(config.DATA_PATH)
        df= pd.read_csv('df2use_train.csv', encoding='latin1')
        df_filt= df[df.num_reviews<=100].reset_index(drop=True)
        if kwargs["products"] == "three":
            asin_list= ['B00008OE43', 'B0007OWASE', 'B000EI0EB8']
        elif kwargs["products"] == "all":
            asin_list= df_filt.asin.tolist()[:]
            # Cap at 1000 products
            asin_list = asin_list[0:1000]
        else:
            raise Exception ("Product group not recognized")
        summary_dict= OrderedDict()
        rouge_list, semantic_score_list, sentiment_score_list = [], [], []
        for i, asin in enumerate(asin_list):
            summary_dict[asin] = {}
            product_reviews= reviews_indexer[asin]
            summary, counts, cosine_score= summarization_module(asin, product_reviews, encoder)
            if len(summary) == 0:
                products_skipped+= 1
                continue
            summary_dict[asin]["summary"]= summary
            rouge_score= rouge_module(summary, product_reviews)
            summary_dict[asin]["rouge"] = rouge_score
            summary_dict[asin]["counts"] = counts
            summary_dict[asin]["cosine_score"] = str(cosine_score)
            rouge_list.append(rouge_score)
            semantic_score_list.append(cosine_score)

            #score is either 1 for correct or 0 for wrong
            reviews_short = reviews_indexer.get_reviews_short(asin)
            assert (len(reviews_short) == len(product_reviews))
           

            reviews_sentiment, summary_sentiment, score_sentiment = evaluate_sentiment (reviews_short, summary, sent_model, sent_tokenizer)
            summary_dict[asin]["reviews_sentiment"] = str(reviews_sentiment)
            summary_dict[asin]["summary_sentiment"] = str(summary_sentiment)
            summary_dict[asin]["sentiment_score"] = str(score_sentiment)

            sentiment_score_list.append (score_sentiment)
            rouge_list.append(rouge_score)
            semantic_score_list.append(cosine_score)
            print (i)
            if i > 0 and i % 50 == 0:
                write_json(kwargs, summary_dict, model)
                print("Rouge metrics")
                print(pd.Series(rouge_list).describe())
                print("Semantic score metrics")
                print(pd.Series(semantic_score_list).describe())
                print("Sentiment score metrics")
                print(pd.Series(sentiment_score_list).describe())
            sys.stdout.flush()
        print(np.mean(rouge_list))
        print("Rouge metrics")
        print(pd.Series(rouge_list).describe())
        print("Semantic score metrics")
        print(pd.Series(semantic_score_list).describe())
        print("Sentiment score metrics")
        print(pd.Series(sentiment_score_list).describe())
        print("Finished run, {} products were skipped due to run-time exceptions".format(products_skipped))
        write_json(kwargs, summary_dict, model)

if args.debug == False:
    pdb.set_trace= lambda:None


if __name__ == "__main__":
   with slaunch_ipdb_on_exception():
       if args.prepare_embeddings == True:
           main_preprocess_embeddings(vars(args))
       else:
           main(vars(args))


# def main_preprocess_embeddings2(kwargs):
#     pdb.set_trace()
#     db_file= os.path.join(config.DATA_PATH, "embedding_db-{}.s3db".format(args.encoder_name))
#     conn= sqlite3.connect(db_file)
#     cur= conn.cursor()
#     cur.execute("drop table if exists reviews_embeddings;")
#     cur.execute("CREATE TABLE IF NOT EXISTS reviews_embeddings ("
#       "asin VARCHAR(255) PRIMARY KEY NOT NULL, "
#       "product_embs VARCHAR(255),"
#       "product_sentences VARCHAR(255),"
#       "sentence_parent VARCHAR(255))")

#     classifier = tf.estimator.Estimator(
#         model_fn= my_model,
#         params= params,
#         config= model_config)



#     preprocess_module= PreprocessEncoder(kwargs['summary_length'], embeddings_preprocessed=False)
#     encoder= get_encoder()
#     df= pd.read_csv('df2use_train.csv', encoding='latin1')
#     df_filt= df[df.num_reviews<=100].reset_index(drop=True)
#     if kwargs["products"] == "three":
#         asin_list= ['B00008OE43', 'B0007OWASE', 'B000EI0EB8']
#     elif kwargs["products"] == "all":
#         asin_list= df_filt.asin.tolist()[:]
#         # Cap at 1000 products
#         asin_list = asin_list[0:1000]
#     else:
#         raise Exception ("Product group not recognized")
#     # reviews_indexer= SQLLiteIndexer(config.DATA_PATH)
#     try:
#         reviews_iterator= SQLLiteAsinAttrIterator(asin_list)
#         values_to_insert= []
#         ddict= defaultdict(list)
#         write= True
#         for i, row_tup in enumerate(reviews_iterator):
#             asin, product_reviews= row_tup
#             product_sentences, product_embs, sentence_parent= preprocess_module(asin, product_reviews, encoder)
#             # product_sentences, product_embs, sentence_parent= product_reviews, np.random.rand(len(product_reviews)*100, 500), np.random.randint(0,1000, 100*len(product_reviews))
#             # insert_emb(cur, asin, product_embs.tolist(), product_sentences, sentence_parent)
#             values_to_insert.append((str(asin), str(product_embs.tolist()), str(product_sentences), str(sentence_parent)))
            
#             # ddict['asin'].append(asin)
#             # ddict['product_embs'].append(product_embs.tolist())
#             # ddict['product_sentences'].append(product_sentences)
#             # ddict['sentence_parent'].append(sentence_parent)
#             logging.info(i)
#             gc.collect()

#             if i > 0 and i % 50 == 0:
#                 insert_emb_many(cur, values_to_insert)
#                 # insert_emb_csv(cur, values_to_insert, write)
#                 write= False
#                 ddict= defaultdict(list)
#                 logging.info("Inserted {} total products to embeddings_db".format(i+1))
#                 values_to_insert= []

#             if i > 0 and i % 500 == 0:
#                 conn.commit()
#                 logging.info("Commited {} total products to embeddings_db".format(i+1))
#                 cur.execute('select count(*) from reviews_embeddings')
#                 logging.info(cur.fetchone())
#                 gc.collect()
#             sys.stdout.flush()
#         pdb.set_trace()
#         insert_emb_many(cur, values_to_insert)
#         conn.commit()
#         logging.info("Finished {} total products to embeddings_db".format(i+1))
#         cur.execute('select count(*) from reviews_embeddings')
#         logging.info(cur.fetchone())
#         pdb.set_trace()
#         test_indexer= SQLLiteEmbeddingsIndexer(args.encoder_name)
#         ddict= test_indexer[asin]
#         assert ddict['asin'] == asin
#         assert ddict['product_sentences'] == product_sentences
#         assert ddict['sentence_parent'] == sentence_parent
#         np.testing.assert_equal(ddict['product_embs'], product_embs)
#     except KeyboardInterrupt:
#         logging.info("Keyboard interrup, commiting remaining")
#         conn.commit()
#     except Exception as e:
#         logging.info("Error type: {}".format(type(e).__name__))
#         conn.rollback()
#     finally:
#         cur.close()
#         conn.close()
