#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 15:05:12 2018

@author: hanozbhathena
"""

import tensorflow as tf
import tensorflow_hub as hub
from nltk.tokenize import word_tokenize

ENCODER_PATH_DICT= {'nnlm': 'https://tfhub.dev/google/nnlm-en-dim128-with-normalization/1',
                    'w2v': 'https://tfhub.dev/google/Wiki-words-500-with-normalization/1',
                    'use': 'https://tfhub.dev/google/universal-sentence-encoder/2',
                    'elmo': 'https://tfhub.dev/google/elmo/2',
                    }


class BaseHubModel(object):
    def __init__(self):
        self.sess= tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.tables_initializer())
    
    def __del__(self):
        self.sess.close()


class NNLM(BaseHubModel):
    def __init__(self, path= ENCODER_PATH_DICT['nnlm'],
                 trainable= False):
        self.model= hub.Module(path, trainable= trainable)
        super().__init__()
    
    def __call__(self, inp):
        return self.sess.run(self.model(inp))


class Word2Vec(BaseHubModel):
    def __init__(self, path= ENCODER_PATH_DICT['w2v'],
                 trainable= False):
        self.model= hub.Module(path, trainable= trainable)
        super().__init__()
    
    def __call__(self, inp):
        return self.sess.run(self.model(inp))


class USE(BaseHubModel):
    max_seq_len= 600
    def __init__(self, path= ENCODER_PATH_DICT['use'],
                 trainable= False):
        self.model= hub.Module(path, trainable= trainable)
        super().__init__()
    
    def __call__(self, inp):
        inp= [' '.join(word_tokenize(sentence)[:self.max_seq_len]) for sentence in inp]
        return self.sess.run(self.model(inp))


class ELMO(BaseHubModel):
    max_seq_len= 600
    def __init__(self, path= ENCODER_PATH_DICT['elmo'],
                 trainable= False):
        self.model= hub.Module(path, trainable= trainable)
        super().__init__()
    
    def __call__(self, inp):
        inp= [' '.join(word_tokenize(sentence)[:self.max_seq_len]) for sentence in inp]
        embeddings= self.model(inp, signature="default", as_dict=True)["elmo"] # (? x max_seq_len x 1024)
        max_pool= tf.reduce_max(embeddings, axis= 1)
        return self.sess.run(max_pool)



