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

from model_data import train_input_fn
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
                average_across_batch=True)
    return seq_loss


def cosine_loss(a, b):
    """
    a: (batch_size*num_products x hidden_size*num_lstm_layers)
    b: (num_products x hidden_size*num_lstm_layers)
    """
    # pdb.set_trace()
    normalize_a = tf.nn.l2_normalize(a, 1)
    normalize_b = tf.nn.l2_normalize(b, 1)
    cosine_similarities= tf.matmul(normalize_a, normalize_b, transpose_b= True) #(batch_size*num_products x num_products)
    cosine_similarities= tf.reshape(cosine_similarities, [-1]) #(batch_size*num_products*num_products, )
    cos_distance= tf.reduce_mean(1. - cosine_similarities) #scalar --> ()
    return cos_distance


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
        product_reviews= tf.split(summar_id_list, params['abs_num_reviews'], axis=0)
        padded_summar_id_list= tf.concat([tf.concat([p]*BATCH_SIZE, axis=0) for p in product_reviews], axis= 0)

        ae_word_ids= ae_decoder_output['decoder_word_ids']
        predictions = {
            'summary_ids': padded_summar_id_list,
            'asin': features['asin_list'],
            'text_list': features['data_batch'],
            'ae_word_ids': ae_word_ids,
            'real_lens': features['real_lens'],
            'ae_logits': ae_decoder_output['decoder_logits'],
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    pdb.set_trace()
    # Compute loss.
    ae_loss= seq_ae_loss(logits= ae_decoder_output['decoder_logits'], 
                         targets= features['data_batch'], real_lens= features['real_lens'])
    # pdb.set_trace()
    ae_encoder_output_c= tf.concat(ae_encoder_output, axis=1) #(batch_size*num_products x hidden_size*num_lstm_layers)
    summ_encoder_output_c= tf.concat(summ_encoder_output, axis=1) #(num_products x hidden_size*num_lstm_layers)
    cos_loss= cosine_loss(ae_encoder_output_c, summ_encoder_output_c)
    
    loss= ae_loss + cos_loss
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


