#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 21:08:42 2018

@author: hanozbhathena
"""


from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import DBSCAN
import dill as pickle
from rouge import Rouge
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import ipdb as pdb
from ipdb import slaunch_ipdb_on_exception
import igraph


class BaseExtract(object):
    def __init__(self, summary_length):
        self.summary_length= summary_length
        

    def reset (self, encoder):
        self.sentence_parent = []
        self.product_sentences = []
        self.counts = []
        self.encoder = encoder
        self.product_embs = None


    def tokenize_and_embed (self, product_reviews):
        for idx, review in enumerate(product_reviews):
            for sent in sent_tokenize(review):
                self.product_sentences.append(sent)
                self.sentence_parent.append(idx)
        # print ("product_reviews_count {}".format(len(product_reviews)))
        # print ("product_sentences_count {}".format(len(self.product_sentences)))

        # print ("Embedding...")
        self.product_embs = self.encoder(self.product_sentences)

    def compute_counts (self, cluster_obj, centroid_labels):
        # pdb.set_trace()
        for label in centroid_labels:
            reviews_for_label = []
            for idx, review_label in enumerate(cluster_obj.labels_):
                if review_label == label:
                    reviews_for_label.append (self.sentence_parent[idx])
            count = len ( set (reviews_for_label))
            self.counts.append (count)
    
    def sematic_similarity(self, product_embs, summary_embs):
        product_mean= product_embs.mean(axis= 0, keepdims=True)
        summary_mean= summary_embs.mean(axis= 0, keepdims=True)
        cosine= cosine_similarity(product_mean, summary_mean)[0][0]
        return cosine        


class KMeansExtract(BaseExtract):
    def __call__(self, product_reviews, encoder):
        self.reset (encoder)
        self.tokenize_and_embed (product_reviews)

        print ("Running kmeans...")
        cluster_obj = KMeans(n_clusters=self.summary_length, random_state=0).fit(self.product_embs)
        
        dist= cluster_obj.transform(self.product_embs)
        product_reviews_np= np.array(self.product_sentences)
        summary_reviews= product_reviews_np[np.argmin(dist, axis=0)].tolist()
        centroid_labels = range(self.summary_length)

        self.compute_counts (cluster_obj, centroid_labels)
        
        summary_embs= self.encoder(summary_reviews)
        cosine_score= self.sematic_similarity(self.product_embs, summary_embs)
        return summary_reviews, self.counts, cosine_score


class AffinityExtract(BaseExtract):   
    def __call__(self, product_reviews, encoder):
        self.reset (encoder)
        self.encoder = encoder
        self.tokenize_and_embed (product_reviews)

        print("Running affinity...")
        cluster_obj = AffinityPropagation().fit(self.product_embs)
        num_clusters = len(cluster_obj.cluster_centers_)

        product_reviews_np= np.array(self.product_sentences)
        cluster_counts = defaultdict(int)
        for label in cluster_obj.labels_:
            cluster_counts[label] += 1
        sorted_by_value = sorted(cluster_counts.items(), key=lambda kv: kv[1], reverse = True)
        #pick the largest cluster_obj
        top_center_indicies = [kv[0] for kv in sorted_by_value][0:self.summary_length]
        summary_indicies = []
        for cluster_center_index in cluster_obj.cluster_centers_indices_:
            label = cluster_obj.labels_[cluster_center_index]
            if label in top_center_indicies:
                summary_indicies.append(cluster_center_index)
        summary_reviews = product_reviews_np[summary_indicies].tolist()
        centroid_labels = top_center_indicies
        self.compute_counts (cluster_obj, centroid_labels)
        
        summary_embs= self.encoder(summary_reviews)
        cosine_score= self.sematic_similarity(self.product_embs, summary_embs)
        return summary_reviews, self.counts, cosine_score


class DBSCANExtract(BaseExtract):  
    def __call__(self, product_reviews, encoder):
        self.reset (encoder)
        self.tokenize_and_embed (product_reviews)

        print("Running dbscan...")
        eps = 0.20131
        cluster_obj = DBSCAN(eps=eps, metric="cosine", min_samples=2)
        cluster_obj.fit(self.product_embs)
        num_clusters = len(set(cluster_obj.labels_))

        product_reviews_np = np.array(self.product_sentences)
        cluster_counts = defaultdict(int)
        for label in cluster_obj.labels_:
            cluster_counts[label] += 1
        sorted_by_value = sorted(cluster_counts.items(), key=lambda kv: kv[1], reverse = True)
        top_center_indicies = [kv[0] for kv in sorted_by_value][0:self.summary_length]
        label_to_summary_index = {}
        for cluster_center_index in cluster_obj.core_sample_indices_:
            label = cluster_obj.labels_[cluster_center_index]
            if label in top_center_indicies and not label in label_to_summary_index:
                 s = self.product_sentences[cluster_center_index]
                 if len(s) > 10 and not '.' in s[0:-2]:
                     label_to_summary_index[label] = cluster_center_index
            if len(label_to_summary_index) >= num_clusters:
                break
        summary_indicies = list(label_to_summary_index.values())
        summary_reviews = product_reviews_np[summary_indicies].tolist()
        centroid_labels = top_center_indicies

        self.compute_counts (cluster_obj, centroid_labels)

        summary_embs= self.encoder(summary_reviews)
        cosine_score= self.sematic_similarity(self.product_embs, summary_embs)
        return summary_reviews, self.counts, cosine_score

        

class PageRankExtract_slow(BaseExtract):
    def __init__(self, *args, **kwargs):
        # pdb.set_trace()
        self.max_summ_sent_len= kwargs.pop('max_summ_sent_len', 30)
        super().__init__(*args, **kwargs)

    def __call__(self, product_reviews, encoder):
        self.reset (encoder)
        self.tokenize_and_embed (product_reviews)
        summary_reviews = []

        print("Running pagerank...")
        
        sim_mat= cosine_similarity(self.product_embs)
        # sim_mat= (cosine_similarity(self.product_embs) + 1)/2
        graph= nx.from_numpy_array(sim_mat)
        try:
            scores= nx.pagerank(graph, max_iter=50)
        except nx.exception.PowerIterationFailedConvergence:
            return summary_reviews, self.counts, -2.0
        ranked_sentences= sorted(((scores[i],s) for i, s in enumerate(self.product_sentences)), reverse=True)

        # loop through the ranked sentences; if the word length is less than threshold add to the summary
        
        summary_len= 0
        ranked_meta= [(len(ranked_sentences), )] #to record the number of sentences to better interpret the rank
        for i in range(len(ranked_sentences)):
            candidate_summary= ranked_sentences[i][1]
            if len(word_tokenize(candidate_summary)) > self.max_summ_sent_len:
                continue
            summary_reviews.append(candidate_summary)
            summary_len+= 1
            # maintain ranked_meta as a tuple of the rank of the ranked_sentence and its word len going into the summary
            ranked_meta.append((i, len(word_tokenize(candidate_summary))))
            # if the number of summaries reaches the threshold of output summaries break
            if summary_len == self.summary_length:
                break

        # this should rarely if ever be called; but it will make sure that there are self.summary_length 
        # number of sentences output as summary if the above for loop doesnt do it
        i=0
        while summary_len < self.summary_length:
            candidate_summary= ranked_sentences[i][1]
            if candidate_summary in summary_reviews:
                continue
            summary_reviews.append(candidate_summary)
            summary_len+= 1
            ranked_meta.append((i, len(word_tokenize(candidate_summary))))

        #TODO: hanoz please add counts, you need to get the best centroids
        #XXX: counts the way its implemented here is not possible to calculate for pagerank as there is no such
        #       thing as a cluster.labels_. Implementing an alternate meaning of counts for Pagerank
        # self.counts = [0] * self.summary_length
        self.counts= ranked_meta

        summary_embs= self.encoder(summary_reviews)
        cosine_score= self.sematic_similarity(self.product_embs, summary_embs)
        # pdb.set_trace()
        return summary_reviews, self.counts, cosine_score


class PageRankExtract(BaseExtract):
    def __init__(self, *args, **kwargs):
        # pdb.set_trace()
        self.max_summ_sent_len= kwargs.pop('max_summ_sent_len', 30)
        super().__init__(*args, **kwargs)

    def __call__(self, product_reviews, encoder):
        self.reset (encoder)
        self.tokenize_and_embed (product_reviews)
        summary_reviews = []

        print("Running pagerank...")
        
        # sim_mat= cosine_similarity(self.product_embs)
        sim_mat= (cosine_similarity(self.product_embs) + 1)/2
        if True:
            sim_mat = sim_mat * (1. - np.eye(len(sim_mat))) # remove self connection
        graph= igraph.Graph.Adjacency((sim_mat > 0).tolist())
        

        try:
            scores= graph.pagerank(weights= sim_mat[sim_mat.nonzero()].tolist(), niter=100)
        except Exception:
            return [], self.counts, -2.0
        ranked_sentences= sorted(((scores[i],s) for i, s in enumerate(self.product_sentences)), reverse=True)

        # loop through the ranked sentences; if the word length is less than threshold add to the summary
        
        summary_len= 0
        ranked_meta= [(len(ranked_sentences), )] #to record the number of sentences to better interpret the rank
        for i in range(len(ranked_sentences)):
            candidate_summary= ranked_sentences[i][1]
            if len(word_tokenize(candidate_summary)) > self.max_summ_sent_len:
                continue
            summary_reviews.append(candidate_summary)
            summary_len+= 1
            # maintain ranked_meta as a tuple of the rank of the ranked_sentence and its word len going into the summary
            ranked_meta.append((i, len(word_tokenize(candidate_summary))))
            # if the number of summaries reaches the threshold of output summaries break
            if summary_len == self.summary_length:
                break

        # this should rarely if ever be called; but it will make sure that there are self.summary_length 
        # number of sentences output as summary if the above for loop doesnt do it
        i=0
        while summary_len < self.summary_length:
            candidate_summary= ranked_sentences[i][1]
            if candidate_summary in summary_reviews:
                continue
            summary_reviews.append(candidate_summary)
            summary_len+= 1
            ranked_meta.append((i, len(word_tokenize(candidate_summary))))

        #TODO: hanoz please add counts, you need to get the best centroids
        #XXX: counts the way its implemented here is not possible to calculate for pagerank as there is no such
        #       thing as a cluster.labels_. Implementing an alternate meaning of counts for Pagerank
        # self.counts = [0] * self.summary_length
        self.counts= ranked_meta

        summary_embs= self.encoder(summary_reviews)
        cosine_score= self.sematic_similarity(self.product_embs, summary_embs)
        # pdb.set_trace()
        return summary_reviews, self.counts, cosine_score


def get_ex_summarizer(model_type, summary_length= 5):
    if model_type == 'kmeans':
        return KMeansExtract(summary_length)
    elif model_type == 'affinity':
        return AffinityExtract(summary_length)
    elif model_type == 'dbscan':
        return DBSCANExtract(summary_length)
    elif model_type == 'pagerank':
        return PageRankExtract(summary_length)
    elif model_type == "pagerank_slow":
        return PageRankExtract_slow(summary_length)
    else:
        raise ValueError("Invalid model type supplied")


class MyRouge(object):
    def __init__(self):
        self.rouge= Rouge()
    
    def __call__(self, summary_list, reviewTexts):
        summariesConcat= ". ".join (summary_list)
        total = 0
        skipped = 0
        for review in reviewTexts:
            if len(review) > 0:
                total += self.rouge.get_scores(summariesConcat, review)[0]["rouge-1"]["f"]
            else:
                skipped += 1
        rougeAvg = total / (len(reviewTexts) - skipped)
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