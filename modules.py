#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 19:51:23 2018

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
import tensorflow as tf
import dill as pickle
import json

from model_data import train_input_fn, MAX_SEQUENCE_LENGTH
from sequence_modules import summarization_model
from config import args
from model_data import BATCH_SIZE

# def seq_ae_loss(logits, targets, real_lens):
#     weights= tf.sequence_mask(real_lens, targets.get_shape().as_list()[-1], dtype= tf.float32)
#     w= tf.unstack(weights, axis= 1)
#     l= tf.unstack(logits, axis= 1)
#     t= tf.unstack(targets, axis= 1)
#     seq_loss= tf.contrib.legacy_seq2seq.sequence_loss_by_example(l, t, w)
#     seq_loss= tf.reduce_mean(seq_loss)
#     return seq_loss


def seq_ae_loss(logits, targets, real_lens):
    weights= tf.sequence_mask(real_lens, targets.get_shape().as_list()[-1], dtype= tf.float32)
    seq_loss = tf.contrib.seq2seq.sequence_loss(
                tf.to_float(logits),
                tf.to_int32(targets),
                weights,
                average_across_timesteps=True,
                average_across_batch=False)
    return tf.reduce_sum(seq_loss)


def cosine_loss(a, b):
    """
    a: (num_products, num_reviews, combined_hidden_size)
    b: (num_products, combined_hidden_size)
    """
    pdb.set_trace()
    b= tf.expand_dims(b, axis= 1)
    normalize_a = tf.nn.l2_normalize(a, 2)
    normalize_b = tf.nn.l2_normalize(b, 2)
    cosine_similarities= tf.matmul(normalize_a, normalize_b, transpose_b= True) #(num_products x num_reviews)
    cosine_distances= 1. - cosine_similarities
    cosine_loss= tf.reduce_sum(tf.reduce_mean(cosine_distances, axis= 1))
    return cosine_loss


def my_model(features, labels, mode, params):
    pdb.set_trace()
    if mode != tf.estimator.ModeKeys.TRAIN:
        params['config']['dropout_keep']= 1.0
    ret= summarization_model(features, mode, params)
    ae_encoder_output= ret['ae_encoder_output']
    ae_decoder_output= ret['ae_decoder_output']
    summ_encoder_output= ret['summ_encoder_output']
    summar_id_list= ret['summar_id_list']
    layers_dict= ret['layers_dict']
    
    
    # Compute predictions.
    if mode == tf.estimator.ModeKeys.PREDICT:
        # pdb.set_trace()
        # batch_size= tf.shape(features['asin_list'])[0]
        # summar_id_list= tf.tile(summar_id_list, [batch_size, 1])
        # summar_id_list= tf.tile(summar_id_list, [params['abs_num_reviews'], 1])
        # product_reviews= tf.split(summar_id_list, params['abs_num_reviews'], axis=0)
        # padded_summar_id_list= tf.concat([tf.concat([p]*BATCH_SIZE, axis=0) for p in product_reviews], axis= 0)

        ae_word_ids= ae_decoder_output['decoder_word_ids']
        predictions = {
            'summary_ids': ret['summar_id_list'],
            'asin': tf.reshape(features['asin_list'], [-1, params['abs_num_reviews']]),
            'text_list': tf.reshape(features['text_list'], [-1, params['abs_num_reviews']]),
            'asin_num_list': tf.reshape(features['asin_num_list'], [-1, params['abs_num_reviews']]),
            'input_word_ids': tf.reshape(features['data_batch'], [-1, params['abs_num_reviews'], MAX_SEQUENCE_LENGTH]),
            'ae_word_ids': tf.reshape(ae_word_ids, [-1, params['abs_num_reviews'], MAX_SEQUENCE_LENGTH]),
            'real_lens': tf.reshape(features['real_lens'], [-1, params['abs_num_reviews']]),
            # 'ae_logits': ae_decoder_output['decoder_logits'],
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    pdb.set_trace()
    # Compute loss.
    ae_loss= seq_ae_loss(logits= ae_decoder_output['decoder_logits'], 
                         targets= features['data_batch'], real_lens= features['real_lens'])
    # pdb.set_trace()
    ae_encoder_output_c= tf.concat(ae_encoder_output, axis=1) #(num_reviews*num_products x hidden_size*num_lstm_layers)
    summ_encoder_output_c= tf.concat(summ_encoder_output, axis=1) #(num_products x hidden_size*num_lstm_layers)
    last_dim= ae_encoder_output_c.get_shape().as_list()[-1]
    ae_encoder_output_c= tf.reshape(ae_encoder_output_c, [-1, params['abs_num_reviews'], last_dim])
    cos_loss= cosine_loss(ae_encoder_output_c, summ_encoder_output_c)
    
    loss= ae_loss + cos_loss #TODO: review cosine loss weightage; potentially add regression loss for AE
    # Compute evaluation metrics.
    metrics = {'ae_loss': ae_loss, 'cos_loss': cos_loss}
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)
    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN
    # optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
    # train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
    gradients, variables = zip(*optimizer.compute_gradients(loss))
    gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
    train_op = optimizer.apply_gradients(zip(gradients, variables), global_step=layers_dict['global_step'])

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


