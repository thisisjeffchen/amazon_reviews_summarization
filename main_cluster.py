#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 14:47:39 2018

@author: hanozbhathena
"""

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

import config
from main_encode import get_encoder
from data_utils import SQLLiteBatchIterator, SQLLiteIndexer
from extractive_summ_modules import get_ex_summarizer, MyRouge
from sentiment_analysis.sentiment_model import CNN
from keras.models import load_model
from sentiment_analysis.data.util import pad_sentence

SENTENCE_LENGTH = 600


def write_json(kwargs, summary_dict, model):
    file_id = model if kwargs["products"] == "three" else model + "-1000"
    with open(config.RESULTS_PATH + 'summary_dict_proposal_{}.json'.format(file_id), 'w') as fo:
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


def main(kwargs):
    # pdb.set_trace()
    if kwargs['extractive_model'] == "all":
        models = ["kmeans", "affinity", "dbscan", "pagerank", "pagerank_slow"]
    else:
        models = [kwargs['extractive_model']]

    sent_model, sent_tokenizer = load_sentiment_evaluator ()

    products_skipped= 0
    for model in models:
        summarization_module= get_ex_summarizer(model_type= model,
                                                summary_length= kwargs['summary_length'])
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
            summary, counts, cosine_score= summarization_module(product_reviews, encoder)
            if len(summary) == 0:
                products_skipped+= 1
                continue
            summary_dict[asin]["summary"]= summary
            rouge_score= rouge_module(summary, product_reviews)
            summary_dict[asin]["rouge"] = rouge_score
            summary_dict[asin]["counts"] = counts
            summary_dict[asin]["cosine_score"] = str(cosine_score)
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

            print(i)
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


if __name__ == "__main__":
   with slaunch_ipdb_on_exception():
       main(vars(config.args))
