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
import config

#from main_encode import get_encoder
#from data_utils import SQLLiteBatchIterator, SQLLiteIndexer
#from text_encoders import ENCODER_PATH_DICT
from sequence_modules import summarization_model


def seq_ae_loss(logits, targets, real_lens):
    weights= tf.sequence_mask(real_lens, targets.get_shape().as_list()[-1], dtype= tf.float32)
    w= tf.unstack(weights, axis= 1)
    l= tf.unstack(logits, axis= 1)
    t= tf.unstack(targets, axis= 1)
    seq_loss= tf.contrib.legacy_seq2seq.sequence_loss_by_example(l, t, w)
    seq_loss= tf.reduce_mean(seq_loss)
    return seq_loss


def cosine_loss(a, b):
    pdb.set_trace()
    normalize_a = tf.nn.l2_normalize(a, 0)
    normalize_b = tf.nn.l2_normalize(b, 0)
    cos_similarity= tf.reduce_sum(tf.multiply(normalize_a,normalize_b))
    cos_loss= 1. - cos_similarity
    return cos_loss


def my_model(features, labels, mode, params):
    #pdb.set_trace()
    ret= summarization_model(features, mode, params)
    ae_encoder_output= ret['ae_encoder_output']
    ae_decoder_output= ret['ae_decoder_output']
    summ_encoder_output= ret['summ_encoder_output']
    summar_text_list= ret['summar_text_list']

    #pdb.set_trace()
    # Compute predictions.
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'summary': summar_text_list,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    # Compute loss.
    ae_loss= seq_ae_loss(logits= ae_decoder_output['decoder_logits'], 
                         targets= features['data_batch'], real_lens= features['real_lens'])
    
    t_summ_encoder_output= tf.tile(summ_encoder_output, [tf.shape(ae_encoder_output)[0], 1])
    cos_loss= cosine_loss(ae_encoder_output, t_summ_encoder_output)
    
    loss= ae_loss + cos_loss
    # Compute evaluation metrics.
    metrics = {'ae_loss': ae_loss, 'cos_loss': cos_loss}
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)
    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN
    optimizer = tf.train.AdagradOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


## Model
def model(features, mode, params):
    is_training = mode == tf.estimator.ModeKeys.TRAIN
    
    # Setup model architecture
    # Enable training of mode == tf.contrib.learn.ModeKeys.TRAIN
    with tf.variable_scope('Input'):
        input_layer = tf.reshape(
            features, 
            shape=[1, 1, 1], 
            name='input_reshape')

    with tf.name_scope('Dense1'):
        model_output = tf.layers.dense(
            inputs=input_layer, 
            units=10, 
            trainable=is_training)
        
        return model_output

## Model Function
# Have to remove type annotations until https://github.com/tensorflow/tensorflow/issues/12249
def custom_model_fn(features, labels, mode, params):
    """Model function used in the estimator.
    Args:
        features (Tensor): Input features to the model.
        labels (Tensor): Labels tensor for training and evaluation.
        mode (ModeKeys): Specifies if training, evaluation or prediction.
        params (HParams): hyperparameters.
    Returns:
        (EstimatorSpec): Model to be run by Estimator.
    """
    model_output = model(features, mode, params)
    
    # Get prediction of model output
    predictions = {
        'classes': tf.argmax(model_output),
        'probabilities': tf.nn.softmax(model_output, name='softmax_tensor')
    }
    
    # PREDICT
    if mode == tf.estimator.ModeKeys.PREDICT:
        export_outputs = {
            'predict_output': tf.estimator.export.PredictOutput(predictions)
        }
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)
    loss = tf.losses.softmax_cross_entropy(
        labels=tf.cast(labels, tf.int32),
        logits=model_output
    )
    # Configure the Training Op (for TRAIN mode)
    if mode == tf.contrib.learn.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=params.learning_rate,
            optimizer=tf.train.AdamOptimizer
        )
        
        # Return an EstimatorSpec object for training
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, loss=loss, train_op=train_op)
    eval_metric = {
        'accuracy': tf.metrics.accuracy(
            labels=tf.cast(labels, tf.int32),
            predictions=model_output,
            name='accuracy'
        )
    }    
    
    # Return a EstimatorSpec object for evaluation
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, loss=loss, eval_metric_ops=eval_metric)



# =============================================================================
# testing
# =============================================================================

def test_model():
    with open('tokenizer.pkl', 'rb') as fi:
        tokenizer= pickle.load(fi)
    
    pdb.set_trace()
    word_emb_size= 300
    params= {}
    params['tokenizer']= tokenizer
    params['token2id']= tokenizer.word_index
    params['vocab_size']= tokenizer.num_words
    params['word_embeddings']= np.load('pretrained_embeddings.npy')
    params['word_embeddings_dim']= word_emb_size
    params['encoder_output_size']= 512
    params['pretrained_encoder']= False
    
    config= {}
    config['num_layers']= 1
    config['hidden_size']= 512
    config['dropout_keep']= 0.9
    
    params['config']= config
    
#    from model_data import test
#    features= test()
    
    #pdb.set_trace()
    mode= 'train'
#    from sequence_modules import summarization_model
#    ret= summarization_model(features, mode, params)
    
    from model_data import train_input_fn
    classifier = tf.estimator.Estimator(
        model_fn= my_model,
        params= params)
    # Train the Model.
    classifier.train(
        input_fn=lambda: train_input_fn(),
        steps=2)


test_model()