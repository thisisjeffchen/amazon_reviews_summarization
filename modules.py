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


def seq_ae_loss(logits, targets, real_lens):
    weights= tf.sequence_mask(real_lens, targets.get_shape().as_list()[-1], dtype= tf.float32)
    w= tf.unstack(weights, axis= 1)
    l= tf.unstack(logits, axis= 1)
    t= tf.unstack(targets, axis= 1)
    seq_loss= tf.contrib.legacy_seq2seq.sequence_loss_by_example(l, t, w)
    seq_loss= tf.reduce_mean(seq_loss)
    return seq_loss


def cosine_loss(a, b):
    """
    a: (batch_size x emb_size) --> (num_reviews x emb_size)
    b: (num_products x emb_size) --> right now is (1 x emb_size)
    """
    # pdb.set_trace()
    normalize_a = tf.nn.l2_normalize(a, 1)
    normalize_b = tf.nn.l2_normalize(b, 1)
    cosine_similarities= tf.squeeze(tf.matmul(normalize_a, normalize_b, transpose_b= True), axis= 1)
    cos_distance= tf.reduce_mean(1. - cosine_similarities)
    return cos_distance


def my_model(features, labels, mode, params):
    #pdb.set_trace()
    if mode != tf.estimator.ModeKeys.TRAIN:
        params['config']['dropout_keep']= 1.0
    ret= summarization_model(features, mode, params)
    ae_encoder_output= ret['ae_encoder_output']
    ae_decoder_output= ret['ae_decoder_output']
    summ_encoder_output= ret['summ_encoder_output']
    summar_id_list= ret['summar_id_list']

    
    # Compute predictions.
    if mode == tf.estimator.ModeKeys.PREDICT:
        # pdb.set_trace()
        predictions = {
            'summary_ids': summar_id_list,
            'asin': tf.expand_dims(features['asin_list'][0], axis= 0),
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    # Compute loss.
    ae_loss= seq_ae_loss(logits= ae_decoder_output['decoder_logits'], 
                         targets= features['data_batch'], real_lens= features['real_lens'])
    # pdb.set_trace()
    # t_summ_encoder_output= tf.tile(summ_encoder_output, [tf.shape(ae_encoder_output)[0], 1])
    cos_loss= cosine_loss(ae_encoder_output, summ_encoder_output)
    
    loss= ae_loss + cos_loss
    # Compute evaluation metrics.
    metrics = {'ae_loss': ae_loss, 'cos_loss': cos_loss}
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)
    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN
    optimizer = tf.train.AdagradOptimizer(learning_rate=params['learning_rate'])
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


