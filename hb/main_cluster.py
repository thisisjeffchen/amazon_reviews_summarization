#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 14:47:39 2018

@author: hanozbhathena
"""

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
from sklearn.cluster import KMeans
import dill as pickle
from rouge import Rouge
import json
from nltk.tokenize import sent_tokenize

DATA_PATH= os.environ['DATA_PATH']
from config import args
from main_encode import get_encoder
from data_utils import SQLLiteBatchIterator, SQLLiteIndexer


#ref= "***".join(product_reviews)
#hyp= "***".join(summary_reviews)
#rouge= Rouge()
#rouge.get_scores(hyp, ref)
#

def get_norm_rouge(summary_list, ground_truth_sentences):
    rouge= Rouge()
    rouge_list= []
    for hyp in summary_list:
        scores= [rouge.get_scores(hyp.lower(), r.lower())[0]['rouge-1']['f'] 
                    for r in ground_truth_sentences if len(r)>10]
        rouge_list.extend(scores)
    return np.mean(rouge_list)


def get_kmeans_summary(product_reviews, encoder):
    product_sentences= [sent for review in product_reviews for sent in sent_tokenize(review)]
    product_embs= encoder(product_sentences)
    kmeans= KMeans(n_clusters=5, random_state=0).fit(product_embs)
    dist= kmeans.transform(product_embs)
    product_reviews_np= np.array(product_sentences)
    summary_reviews= product_reviews_np[np.argmin(dist, axis=0)].tolist()
    score= get_norm_rouge(summary_reviews, product_reviews_np.tolist())
    return summary_reviews, score


def main():
    reviews_indexer= SQLLiteIndexer(DATA_PATH)
    encoder= get_encoder()
#    asin_list= ['B00001WRSJ', 'B009AYLDSU', 'B007I5JT4S']
    df= pd.read_csv('df2use_train.csv', encoding='latin1')
    df_filt= df[df.num_reviews<=60].reset_index(drop=True)
    asin_list= df_filt.asin.tolist()[:100]
    asin_list= ['B00008OE43', 'B0007OWASE', 'B000EI0EB8']
    summary_dict= OrderedDict()
    rouge_list= []
    for i, asin in enumerate(asin_list):
#        pdb.set_trace()
        product_reviews= reviews_indexer[asin]
        summary_dict[asin], rouge_score= get_kmeans_summary(product_reviews, encoder)
        rouge_list.append(rouge_score)
        print(i)
    
    print(np.mean(rouge_list))
    print(pd.Series(rouge_list).describe())
    
    with open('summary_dict_proposal.json', 'w') as fo:
        json.dump(summary_dict, fo)


if __name__ == "__main__":
    with slaunch_ipdb_on_exception():
        main()
