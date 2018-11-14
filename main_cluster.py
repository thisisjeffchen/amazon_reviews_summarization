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
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import DBSCAN
import dill as pickle
from rouge import Rouge
import json
from nltk.tokenize import sent_tokenize
import config
from main_encode import get_encoder
from data_utils import SQLLiteBatchIterator, SQLLiteIndexer


#ref= "***".join(product_reviews)
#hyp= "***".join(summary_reviews)
#rouge= Rouge()
#rouge.get_scores(hyp, ref)
#

def get_norm_rouge1(summary_list, ground_truth_sentences):
    rouge= Rouge()
    rouge_list= []
    for hyp in summary_list:
        scores= [rouge.get_scores(hyp.lower(), r.lower())[0]['rouge-1']['f'] 
                    for r in ground_truth_sentences if len(r)>10]
        rouge_list.extend(scores)
    return np.mean(rouge_list)


def get_norm_rouge2(summary_list, reviewTexts):
    rouge= Rouge()
    summariesConcat= ". ".join (summary_list)
    total= 0
    for review in reviewTexts:
      total += rouge.get_scores (summariesConcat, review)[0]["rouge-1"]["f"]
    rougeAvg = total / len(reviewTexts)
    return rougeAvg


def get_norm_rouge3(summary_list, ground_truth_sentences):
    rouge= Rouge()
    summariesConcat= ". ".join (summary_list)
    rouge_list= [rouge.get_scores(summariesConcat.lower(), r.lower())[0]['rouge-1']['f'] 
                for r in ground_truth_sentences if len(r)>10]
    return np.mean(rouge_list)

def get_kmeans_summary(product_reviews, encoder, model="kmeans"):
    sentence_parent = []
    product_sentences = []
    for idx, review in enumerate(product_reviews):
      for sent in sent_tokenize(review):
        product_sentences.append(sent)
        sentence_parent.append(idx)

    counts = []

    print ("product_reviews_count {}".format(len(product_reviews)))
    print ("product_sentences_count {}".format(len(product_sentences)))

    print ("Embedding...")
    product_embs= encoder(product_sentences)


    if model == "kmeans":
        print ("Running kmeans...")
        clusters = KMeans(n_clusters=5, random_state=0).fit(product_embs)
        num_clusters = config.args.num_clusters
    elif model == "affinity":
        print("Running affinity...")
        clusters = AffinityPropagation().fit(product_embs)
        num_clusters = len(clusters.cluster_centers_)
    elif model == "dbscan":
        eps = 0.22
        clusters = DBSCAN(eps=eps, metric="cosine", min_samples=2)
        clusters.fit(product_embs)
        num_clusters = len(set(clusters.labels_))
        print("\nNUM CLUSTERS DBSCAN: %d " % num_clusters)

    if model =="kmeans":
        dist= clusters.transform(product_embs)
        product_reviews_np= np.array(product_sentences)
        summary_reviews= product_reviews_np[np.argmin(dist, axis=0)].tolist()
        centroid_labels = range(num_clusters)
    elif model == "affinity":
        product_reviews_np= np.array(product_sentences)
        cluster_counts = defaultdict(int)
        for label in clusters.labels_:
            cluster_counts[label] += 1
        sorted_by_value = sorted(cluster_counts.items(), key=lambda kv: kv[1], reverse = True)
        #pick the largest clusters
        top_center_indicies = [kv[0] for kv in sorted_by_value][0:config.args.num_clusters]
        summary_reviews= product_reviews_np[top_center_indicies].tolist()
        centroid_labels = top_center_indicies
    elif model == "dbscan":
        product_reviews_np = np.array(product_sentences)
        cluster_counts = defaultdict(int)
        for label in clusters.labels_:
            cluster_counts[label] += 1

        sorted_by_value = sorted(cluster_counts.items(), key=lambda kv: kv[1], reverse = True)
        top_center_indicies = [kv[0] for kv in sorted_by_value][0:config.args.num_clusters]

        summary_indicies = []
        for cluster_center_index in clusters.cluster_centers_indices_:
            label = clusters.labels_[cluster_center_index]
            if label in top_center_indicies:
                summary_indicies.append(cluster_center_index)
        summary_reviews = product_reviews_np[summary_indicies].tolist()
        centroid_labels = top_center_indicies


    for label in centroid_labels:
      reviews_for_label = []
      for idx, review_label in enumerate(clusters.labels_):
        if review_label == label:
          reviews_for_label.append (sentence_parent[idx])
      count = len ( set (reviews_for_label))
      counts.append (count)     

#    score= get_norm_rouge1(summary_reviews, product_reviews_np.tolist())
    if summary_reviews:
        score= get_norm_rouge2(summary_reviews, product_reviews) #rouge2 score does score after concat
#    score= get_norm_rouge3(summary_reviews, product_reviews_np.tolist())
    return summary_reviews, score, counts


def main():
    reviews_indexer= SQLLiteIndexer(config.DATA_PATH)
    encoder= get_encoder()
#    asin_list= ['B00001WRSJ', 'B009AYLDSU', 'B007I5JT4S']
    #df= pd.read_csv('df2use_train.csv', encoding='latin1')
    #df_filt= df[df.num_reviews<=100].reset_index(drop=True)
    #asin_list= df_filt.asin.tolist()[:100]
    asin_list= ['B00008OE43', 'B0007OWASE', 'B000EI0EB8']
    summary_dict= OrderedDict()
    #TODO: we should really put algos in classes and make runners
    for model in ["kmeans", "affinity", "dbscan"]:
    #for model in ["dbscan"]:
        rouge_list= []

        for i, asin in enumerate(asin_list):
#            pdb.set_trace()
            product_reviews= reviews_indexer[asin]
            summary_dict[asin] = {}
            summary, rouge_score, counts = get_kmeans_summary(product_reviews, encoder, model)
            summary_dict[asin]["summary"] = summary
            summary_dict[asin]["rouge"] = rouge_score
            summary_dict[asin]["counts"] = counts
            rouge_list.append(rouge_score)
            print(i)
        
        print(np.mean(rouge_list))
        print(pd.Series(rouge_list).describe())

        
        with open(config.RESULTS_PATH + 'summary_dict_proposal_{}.json'.format(model), 'w') as fo:
            json.dump(summary_dict, fo, ensure_ascii=False, indent=2)
    

def read_file(filename, num_products= 3, num_sents= 5):
    sentences= []
    count= 0
    asins= []
    ret_dict= {}
    with open(filename, 'r') as f:
        for i in range(num_products):
            asin= f.readline().replace('\n', '')
            asins.append(asin)
            for j in range(num_sents):
                sentences.append(f.readline().replace('\n', ''))
            _= f.readline()
            ret_dict[asin]= sentences
            sentences= []
    return ret_dict


def oracle():
    reviews_indexer= SQLLiteIndexer(config.DATA_PATH)
    path= '../github/cs221_project/data/oracle'
    files= ['will.txt', 'jeff.txt']
    review_counts= [3,3]
    rouge_list= []
    for i, file in enumerate(files):
        full_name= os.path.join(path, file)
        asin_dict= read_file(full_name)
        for asin, human_summaries in asin_dict.items():
            product_reviews= reviews_indexer[asin]
            ground_truth_sentences= [sent for review in product_reviews for sent in sent_tokenize(review)]
            score= get_norm_rouge(human_summaries, ground_truth_sentences)
            rouge_list.append(score)
            print(i)
    
    print(np.mean(rouge_list))
    print(pd.Series(rouge_list).describe())




if __name__ == "__main__":
    with slaunch_ipdb_on_exception():
        main()
