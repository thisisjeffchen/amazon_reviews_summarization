#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 21:08:42 2018

@author: hanozbhathena
"""


from sklearn.cluster import KMeans
import dill as pickle
from rouge import Rouge
import numpy as np
from nltk.tokenize import sent_tokenize
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity


class KMeansExtract(object):
    def __init__(self, summary_length):
        self.summary_length= summary_length
    
    def __call__(self, product_reviews, encoder):
        summary_reviews= []
        product_sentences= [sent for review in product_reviews for sent in sent_tokenize(review)]
        product_embs= encoder(product_sentences)
        kmeans= KMeans(n_clusters= self.summary_length, random_state=0).fit(product_embs)
        dist= kmeans.transform(product_embs)
        product_reviews_np= np.array(product_sentences)
        summary_reviews= product_reviews_np[np.argmin(dist, axis=0)].tolist()
        return summary_reviews


class AffinityExtract(object):
    def __init__(self, summary_length):
        self.summary_length= summary_length
    
    def __call__(self, product_reviews, encoder):
        summary_reviews= []
        raise NotImplementedError("To implement")
        return summary_reviews


class DBSCANExtract(object):
    def __init__(self, summary_length):
        self.summary_length= summary_length
    
    def __call__(self, product_reviews, encoder):
        summary_reviews= []
        raise NotImplementedError("To implement")
        return summary_reviews


class PageRankExtract(object):
    def __init__(self, summary_length):
        self.summary_length= summary_length
    
    def __call__(self, product_reviews, encoder):
        summary_reviews= []
        
        product_sentences= [sent for review in product_reviews for sent in sent_tokenize(review)]
        product_embs= encoder(product_sentences)
        
        sim_mat= cosine_similarity(product_embs)
        graph= nx.from_numpy_array(sim_mat)
        try:
            scores= nx.pagerank(graph)
        except nx.exception.PowerIterationFailedConvergence:
            return []
        ranked_sentences= sorted(((scores[i],s) for i, s in enumerate(product_sentences)), reverse=True)
        for i in range(self.summary_length):
            summary_reviews.append(ranked_sentences[i][1])
        return summary_reviews


def get_ex_summarizer(model_type, summary_length= 5):
    if model_type == 'kmeans':
        return KMeansExtract(summary_length)
    elif model_type == 'affinity':
        return AffinityExtract(summary_length)
    elif model_type == 'dbscan':
        return DBSCANExtract(summary_length)
    elif model_type == 'pagerank':
        return PageRankExtract(summary_length)
    else:
        raise ValueError("Invalid model type supplied")


class MyRouge(object):
    def __init__(self):
        self.rouge= Rouge()
    
    def __call__(self, summary_list, reviewTexts):
        summariesConcat= ". ".join (summary_list)
        total= 0
        for review in reviewTexts:
            total += self.rouge.get_scores (summariesConcat, review)[0]["rouge-1"]["f"]
            rougeAvg = total / len(reviewTexts)
        return rougeAvg


# def get_pg_summary(product_reviews, encoder):
#     product_sentences= [sent for review in product_reviews for sent in sent_tokenize(review)]
#     product_embs= encoder(product_sentences)
    
#     sim_mat= cosine_similarity(product_embs)
#     graph= nx.from_numpy_array(sim_mat)
#     try:
#         scores= nx.pagerank(graph)
#     except nx.exception.PowerIterationFailedConvergence:
#         return [], 0.0
#     ranked_sentences= sorted(((scores[i],s) for i, s in enumerate(product_sentences)), reverse=True)
#     summary_reviews= []
#     for i in range(5):
#         summary_reviews.append(ranked_sentences[i][1])
    
#     score= get_norm_rouge2(summary_reviews, product_reviews)
#     return summary_reviews, score


# def get_kmeans_summary(product_reviews, encoder):
#     product_sentences= [sent for review in product_reviews for sent in sent_tokenize(review)]
#     product_embs= encoder(product_sentences)
#     kmeans= KMeans(n_clusters=5, random_state=0).fit(product_embs)
#     dist= kmeans.transform(product_embs)
#     product_reviews_np= np.array(product_sentences)
#     summary_reviews= product_reviews_np[np.argmin(dist, axis=0)].tolist()
#     score= get_norm_rouge2(summary_reviews, product_reviews)
#     return summary_reviews, score


#def read_file(filename, num_products= 3, num_sents= 5):
#    sentences= []
#    count= 0
#    asins= []
#    ret_dict= {}
#    with open(filename, 'r') as f:
#        for i in range(num_products):
#            asin= f.readline().replace('\n', '')
#            asins.append(asin)
#            for j in range(num_sents):
#                sentences.append(f.readline().replace('\n', ''))
#            _= f.readline()
#            ret_dict[asin]= sentences
#            sentences= []
#    return ret_dict
#
#
#def oracle():
#    reviews_indexer= SQLLiteIndexer(DATA_PATH)
#    path= '../github/cs221_project/data/oracle'
#    files= ['will.txt', 'jeff.txt']
#    review_counts= [3,3]
#    rouge_list= []
#    for i, file in enumerate(files):
#        full_name= os.path.join(path, file)
#        asin_dict= read_file(full_name)
#        for asin, human_summaries in asin_dict.items():
#            product_reviews= reviews_indexer[asin]
#            ground_truth_sentences= [sent for review in product_reviews for sent in sent_tokenize(review)]
#            score= get_norm_rouge2(human_summaries, ground_truth_sentences)
#            rouge_list.append(score)
#            print(i)
#    
#    print(np.mean(rouge_list))
#    print(pd.Series(rouge_list).describe())