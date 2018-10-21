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


def get_kmeans_summary(product_reviews, encoder):
    product_sentences= [sent for review in product_reviews for sent in sent_tokenize(review)]
    product_embs= encoder(product_sentences)
    kmeans= KMeans(n_clusters=5, random_state=0).fit(product_embs)
    dist= kmeans.transform(product_embs)
    product_reviews_np= np.array(product_sentences)
    summary_reviews= product_reviews_np[np.argmin(dist, axis=0)].tolist()
    return summary_reviews


def main():
    reviews_indexer= SQLLiteIndexer(DATA_PATH)
    encoder= get_encoder()
    asin_list= ['B00001WRSJ', 'B007I5JT4S']
    summary_dict= OrderedDict()
    
    for i, asin in enumerate(asin_list):
        product_reviews= reviews_indexer[asin]
        summary_dict[asin]= get_kmeans_summary(product_reviews, encoder)
    
    with open('summary_dict.json', 'w') as fo:
        json.dump(summary_dict, fo)
    

if __name__ == "__main__":
    with slaunch_ipdb_on_exception():
        main()
