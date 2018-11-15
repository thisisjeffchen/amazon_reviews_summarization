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
import dill as pickle
import json
from nltk.tokenize import sent_tokenize

import config
from main_encode import get_encoder
from data_utils import SQLLiteBatchIterator, SQLLiteIndexer
from extractive_summ_modules import get_ex_summarizer, MyRouge


def main(kwargs):
    if kwargs['extractive_model'] == "all":
        models = ["kmeans", "affinity", "dbscan", "pagerank"]
    else:
        models = [kwargs['extractive_model']]

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
        else:
            raise Exception ("Product group not recognized")
        summary_dict= OrderedDict()
        rouge_list= []
        for i, asin in enumerate(asin_list):
            summary_dict[asin] = {}
            product_reviews= reviews_indexer[asin]
            summary, counts= summarization_module(product_reviews, encoder)
            if len(summary) == 0:
                continue
            summary_dict[asin]["summary"]= summary
            rouge_score= rouge_module(summary, product_reviews)
            summary_dict[asin]["rouge"] = rouge_score
            summary_dict[asin]["counts"] = counts
            rouge_list.append(rouge_score)
            print(i)
        print(np.mean(rouge_list))
        print(pd.Series(rouge_list).describe())
        
        with open(config.RESULTS_PATH + 'summary_dict_proposal_{}.json'.format(model), 'w') as fo:
            json.dump(summary_dict, fo, ensure_ascii=False, indent=2)


if __name__ == "__main__":
   with slaunch_ipdb_on_exception():
       main(vars(config.args))
