#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 21:20:10 2018

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
import tensorflow_hub as hub
import dill as pickle
import copy

from tensorflow.python.client import device_lib
from config import DATA_PATH, args
from text_encoders import ENCODER_PATH_DICT
from model_data import MAX_SEQUENCE_LENGTH

# =============================================================================
# Globally load the encoder to prevent multiple reloads
# =============================================================================

# https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def sample_gumbel(shape, eps=1e-20): 
    """Sample from Gumbel(0, 1)"""
    U = tf.random_uniform(shape,minval=0,maxval=1)
    return -tf.log(-tf.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature): 
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(tf.shape(logits))
    return tf.nn.softmax( y / temperature)

def gumbel_softmax(logits, temperature, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
        logits: [batch_size, n_class] unnormalized log-probs
        temperature: non-negative scalar
        hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
        [batch_size, n_class] sample from the Gumbel-Softmax distribution.
        If hard=True, then the returned sample will be one-hot, otherwise it will
        be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        k = tf.shape(logits)[-1]
        #y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)
        y_hard = tf.cast(tf.equal(y,tf.reduce_max(y,1,keep_dims=True)),y.dtype)
        y = tf.stop_gradient(y_hard - y) + y
    return y


def construct_cells(config, basic_cell, bidirectional= False):
    def cell():
        return basic_cell(config['hidden_size'])
        # return tf.nn.rnn_cell.DropoutWrapper(basic_cell(config['hidden_size']),
        #                output_keep_prob=config['dropout_keep'],
        #                dtype= tf.float32)
    if bidirectional == True:
        multi_cell_f= tf.contrib.rnn.MultiRNNCell(
                [cell() for _ in range(config['num_layers'])],
                state_is_tuple=True)
        multi_cell_b= tf.contrib.rnn.MultiRNNCell(
                [cell() for _ in range(config['num_layers'])],
                state_is_tuple=True)
        cells= {'fwd': multi_cell_f,
                'bwd': multi_cell_b,
                }
    else:
        multi_cell_f= tf.contrib.rnn.MultiRNNCell(
                [cell() for _ in range(config['num_layers'])],
                state_is_tuple=True)
        cells= {'fwd': multi_cell_f,
                }
    return cells


def hub_seq_encoder():
    emb= encoder(text_list)
    layers= [tf.layers.Dense(output_dim,
                         activation=None,
                         use_bias=False,
                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                         bias_initializer=tf.zeros_initializer())
    ]
    for i, layer in enumerate(layers):
        emb= layer(emb)
    return emb


class custom_init(tf.keras.initializers.Initializer):
    def __init__(self, weight):
        self.weight= tf.convert_to_tensor(weight, dtype= tf.float32)
    
    def __call__(self, shape, dtype=None, partition_info=None):
        return self.weight

class BaseEncoder(object):
    def __init__(self, embedding_layer, projection_layer, config, 
                 enc_cell= None, emb_wts= None, hstate_max= False):
        self.enc_cell= enc_cell
        self.embedding_layer= embedding_layer
        self.projection_layer= projection_layer
        self.config= copy.deepcopy(config)
        self.hstate_max= hstate_max
    
    def __call__(self, final_encoder_state, targets_wids):
        raise NotImplementedError("Overload with teacher forcing or inference mode")


class SeqEncoder(BaseEncoder):
    def __call__(self, inputs, real_seq_lens, apply_embedding_layer= True):
        pdb.set_trace()
        if apply_embedding_layer:
            enc_inputs= self.embedding_layer(inputs)
        else:
            enc_inputs= inputs
        
        if len(self.enc_cell) == 2:
            init_fwd= self.enc_cell['fwd'].zero_state(tf.shape(enc_inputs)[0], tf.float32)
            init_bwd= self.enc_cell['bwd'].zero_state(tf.shape(enc_inputs)[0], tf.float32)
            ((outf, outb), (statef, stateb))= tf.nn.bidirectional_dynamic_rnn(
                                                self.enc_cell['fwd'], self.enc_cell['bwd'],
                                                enc_inputs, sequence_length= real_seq_lens,
                                                initial_state_fw= init_fwd, initial_state_bw= init_bwd,
                                                scope= "encoder")
            out= tf.concat([outf, outb], axis= 2)
            # fstate= tf.concat([statef, stateb], axis= 1)
            num_layers= len(statef)
            fstate= tuple(tf.concat([statef[i], stateb[i]], axis= 1) for i in range(num_layers))
        else:
            init_enc_state= self.enc_cell['fwd'].zero_state(tf.shape(enc_inputs)[0], tf.float32)
            out, fstate= tf.nn.dynamic_rnn(self.enc_cell['fwd'], enc_inputs, 
                                          sequence_length= real_seq_lens,
                                          initial_state= init_enc_state,
                                          scope= "encoder")
        enc_output= tuple(self.projection_layer(s) for s in fstate)
        return enc_output


class PretrainedEncoder(BaseEncoder):
    def __init__(self, *all_args, **kwargs):
        self.encoder= hub.Module(ENCODER_PATH_DICT[args.encoder_name])
        super().__init__(*all_args, **kwargs)
    
    def __call__(self, text_list):
#        pdb.set_trace()
        emb= self.encoder(text_list)
        enc_output= self.projection_layer(emb)
        return enc_output


class BaseDecoder(object):
    def __init__(self, dec_cell, embedding_layer, vocab_softmax_layer, temperature= None):
        self.dec_cell= dec_cell
        self.vocab_softmax= vocab_softmax_layer
        self.embedding_layer= embedding_layer
        self.embedding_layer_wt= embedding_layer.weights[0]
        self.temperature= temperature
    
    def __call__(self, final_encoder_state, targets_wids):
        raise NotImplementedError("Overload with teacher forcing or inference mode")
    
    def infer_word_embedding(self, word_id):
#        return tf.nn.embedding_lookup(self.emb_wts, word_id)
        return self.embedding_layer(word_id)
    
    def argmax_output_word(self, state_vector):
        vocab_vector= self.vocab_softmax(state_vector)
        word_ids= tf.argmax(vocab_vector, axis= -1, output_type= tf.int32)
        return word_ids


class TeacherForcingDecoder_old(BaseDecoder):
    def __call__(self, init_decoder_input, targets_wids):
        #pdb.set_trace()
        dec_state= self.dec_cell.zero_state(tf.shape(init_decoder_input)[0], tf.float32)
        dec_input_ids_list= tf.unstack(targets_wids, axis= 1)
        logits_list= []
        for step, dec_inp_id in enumerate(dec_input_ids_list):
            if step == 0:
                inp_emb= init_decoder_input
            else:
#                inp_emb= self.infer_word_embedding(dec_inp_id)
                inp_emb= self.embedding_layer(dec_inp_id)
            cell_output, dec_state= self.dec_cell(inp_emb, dec_state)
            logits_list.append(self.vocab_softmax(cell_output))
        teacher_forcing_logits= tf.stack(logits_list, axis= 1)
        decoder_word_ids= tf.argmax(teacher_forcing_logits, axis= -1, output_type=tf.int32)
        ret_dict= {'decoder_logits': teacher_forcing_logits,
                   'decoder_word_ids': decoder_word_ids}
        return ret_dict


class TeacherForcingDecoder(BaseDecoder):
    def __call__(self, init_decoder_input, targets_wids, real_seq_lens= None):
        pdb.set_trace()
        # dec_state= self.dec_cell.zero_state(tf.shape(init_decoder_input)[0], tf.float32)
        go_tokens= tf.expand_dims(tf.tile([0], [tf.shape(targets_wids)[0]]), 1)
        decoder_input_ids= tf.concat([go_tokens, targets_wids[:,:-1]], axis= 1)
        decoder_input_embs= self.embedding_layer(decoder_input_ids)
        init_dec_state= init_decoder_input
        out, _= tf.nn.dynamic_rnn(self.dec_cell, decoder_input_embs, 
                                        sequence_length= real_seq_lens,
                                        initial_state= init_dec_state,
                                        scope= "decoder")
        teacher_forcing_logits= self.vocab_softmax(out)
        decoder_word_ids= tf.argmax(teacher_forcing_logits, axis= -1, output_type=tf.int32)
        ret_dict= {'decoder_logits': teacher_forcing_logits,
                   'decoder_word_ids': decoder_word_ids}
        return ret_dict


class InferenceDecoder(BaseDecoder):
    def __call__(self, init_decoder_input, seq_len= 100):
        #pdb.set_trace()
        # dec_state= self.dec_cell.zero_state(tf.shape(init_decoder_input)[0], tf.float32)
        dec_state= init_decoder_input #TODO: correct naming to state once finalized
        inp_emb= tf.zeros([tf.shape(init_decoder_input[0])[0], self.embedding_layer_wt.shape[1]])
        dec_inp_id= None
        logits_list, word_id_list= [], []
        for step in range(seq_len):
            if step == 0:
                pass
                # inp_emb= init_decoder_input
            else:
                inp_emb= self.embedding_layer(dec_inp_id)
            cell_output, dec_state= self.dec_cell(inp_emb, dec_state)
            logits_list.append(self.vocab_softmax(cell_output))
            dec_inp_id= self.argmax_output_word(cell_output)
            word_id_list.append(dec_inp_id)
        inf_decoder_logits= tf.stack(logits_list, axis= 1)
        decoder_word_ids= tf.stack(word_id_list, axis= 1)
        ret_dict= {'decoder_logits': inf_decoder_logits,
                   'decoder_word_ids': decoder_word_ids}
        return ret_dict


class GumbelSoftmaxDecoder(BaseDecoder):

    def __call__(self, init_decoder_input, seq_len= 100):
        """
        Must return a dictionary ret_dict with keys 'decoder_logits' and 'decoder_word_ids'
        of size (?, L, vocab_size) and (?, L) and types tf.float32 and tf.int32 respectively
        """
        pdb.set_trace()
        # dec_state= self.dec_cell.zero_state(tf.shape(init_decoder_input)[0], tf.float32)
        dec_state= init_decoder_input #TODO: correct naming to state once finalized
        next_word_embedding= tf.zeros([tf.shape(init_decoder_input[0])[0], self.embedding_layer_wt.shape[1]])
        # next_word_embedding= tf.identity(init_decoder_input)
        gumbel_softmax_logits= None
        logits_list, word_id_list, word_emb_list= [], [], []
        for step in range(seq_len):
            inp_emb= next_word_embedding
            cell_output, dec_state= self.dec_cell(inp_emb, dec_state)
            vocab_logits= self.vocab_softmax(cell_output)
            gumbel_softmax_logits= gumbel_softmax(vocab_logits, self.temperature)
            logits_list.append(gumbel_softmax_logits)
            dec_inp_id= self.argmax_output_word(cell_output)
            word_id_list.append(dec_inp_id)
            next_word_embedding= tf.matmul(gumbel_softmax_logits, self.embedding_layer_wt)
            word_emb_list.append(next_word_embedding)

        inf_decoder_logits= tf.stack(logits_list, axis= 1)
        decoder_word_ids= tf.stack(word_id_list, axis= 1)
        decoder_word_embeddings= tf.stack(word_emb_list, axis= 1)
        ret_dict= {'decoder_logits': inf_decoder_logits,
                   'decoder_word_ids': decoder_word_ids,
                   'decoder_word_embeddings': decoder_word_embeddings,
                   }
        return ret_dict


def seq2seq_ae(features, mode, params, layers_dict):
    #pdb.set_trace()
    is_train= mode == tf.contrib.learn.ModeKeys.TRAIN
    
    encoder= layers_dict['encoder']
    if params['pretrained_encoder'] == True:
        ae_encoder_output= encoder(features['text_list']) #(batch_size x encoder_emb_size)
    else:
        ae_encoder_output= encoder(features['data_batch'], features['real_lens']) #(batch_size x encoder_emb_size)
    
    ae_decoder_output= None
    # if is_train == True:
    if True: #TODO: Potentially remove this for inference when model is finalized, right now test time inference calls this AE word output
        decoder= TeacherForcingDecoder(layers_dict['dec_cell'], layers_dict['embedding_layer'], 
                           layers_dict['vocab_softmax_layer'])
        ae_decoder_output= decoder(init_decoder_input= ae_encoder_output, targets_wids= features['data_batch'], real_seq_lens= features['real_lens'])
    
    return ae_encoder_output, ae_decoder_output


def decode_id_to_string(word_ids, params):
    word_ids= tf.cast(word_ids, tf.int64)
    tokenizer= params['tokenizer']
    keys= tf.constant(list(tokenizer.index_word.keys()), dtype=tf.int64)
    values= tf.constant(list(tokenizer.index_word.values()), dtype=tf.string)
    table= tf.contrib.lookup.HashTable(
                tf.contrib.lookup.KeyValueTensorInitializer(keys, values, tf.int64, tf.string), "")
    tok_string= table.lookup(word_ids) #eg [b'so', b'far', b'two'...]
    tok_string2= tf.split(tok_string, tok_string.get_shape().as_list()[0]) #eg [[b'so'], [b'far'], [b'two']...]
    sent_string= tf.strings.join(tok_string2, ' ')
    return sent_string


def summarizer(features, mode, params, layers_dict):
    pdb.set_trace()
    is_train= mode == tf.contrib.learn.ModeKeys.TRAIN
    
    ae_encoder_output= features['ae_encoder_output'] # (batch_size x word_emb_size)
    # decoder_input= tf.reduce_mean(ae_encoder_output, axis= 0, keepdims=True) # TODO: grouby mean asin using tf.math.segment_mean
    # decoder_input= tuple(tf.math.segment_mean(s, features['asin_num_list']) for s in ae_encoder_output)
    ae_encoder_output_r= tuple(tf.reshape(s, [-1, params['abs_num_reviews'], params['config']['hidden_size']]) for s in ae_encoder_output)
    decoder_input= tuple(tf.reduce_mean(s, axis=1) for s in ae_encoder_output_r)

    if not is_train:
        # TODO: check problem with InferenceDecoder and change back from GumbelSoftmaxDecoder now
        decoder= InferenceDecoder(layers_dict['dec_cell'], layers_dict['embedding_layer'], 
                           layers_dict['vocab_softmax_layer'])
        summ_decoder_output= decoder(init_decoder_input= decoder_input, seq_len= MAX_SEQUENCE_LENGTH)
        # decoder= GumbelSoftmaxDecoder(layers_dict['dec_cell'], layers_dict['embedding_layer'], 
        #                    layers_dict['vocab_softmax_layer'], temperature= layers_dict['temperature'])
        # summ_decoder_output= decoder(init_decoder_input= decoder_input, seq_len= MAX_SEQUENCE_LENGTH)
    else:
        decoder= GumbelSoftmaxDecoder(layers_dict['dec_cell'], layers_dict['embedding_layer'], 
                           layers_dict['vocab_softmax_layer'], temperature= layers_dict['temperature'])
        summ_decoder_output= decoder(init_decoder_input= decoder_input, seq_len= MAX_SEQUENCE_LENGTH)
    
    pdb.set_trace()
    encoder= layers_dict['encoder']
    summary_wids= summ_decoder_output['decoder_word_ids']
    summ_encoder_output= None
    if is_train: #TODO: bring back when finalizing
    # if True:
        if params['pretrained_encoder'] == True:
            raise ValueError("Cant use pretrained encoder for this as input to encoder is in word embeddings ONLY!")
        else:
            encoder_inputs= summ_decoder_output['decoder_word_embeddings']
            summ_encoder_output= encoder(encoder_inputs, None, apply_embedding_layer= False)
    # else:
    #     if False:
    #         encoder_inputs= summ_decoder_output['decoder_word_embeddings']
    #         summ_encoder_output= encoder(encoder_inputs, None, apply_embedding_layer= False)
    #     else:
    #         summ_encoder_output= None
        # summar_text_list= decode_id_to_string(tf.reshape(summary_wids, (-1,)), params) #length 1 list
    
    return summ_encoder_output, summary_wids


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def build_layers(features, mode, params):
    pdb.set_trace()
    def get_cell():
        if len(get_available_gpus()) > 0:
            return tf.nn.rnn_cell.GRUCell
        else:
            return tf.contrib.rnn.GRUBlockCellV2
    
    layers_dict= {}
    vocab_size= params['vocab_size']
    try:
        init_embeddings= tf.keras.initializers.Constant(params['word_embeddings'])
        embedding_layer= tf.keras.layers.Embedding(input_dim= params['word_embeddings'].shape[0], 
                        output_dim= params['word_embeddings'].shape[1], embeddings_initializer= init_embeddings,
                        trainable= False)
    except KeyError:
        init_embeddings= 'uniform'
        embedding_layer= tf.keras.layers.Embedding(input_dim= vocab_size+1, 
                        output_dim= params['word_embeddings_dim'], embeddings_initializer= init_embeddings)
    layers_dict['embedding_layer']= embedding_layer

    if params['tie_in_out_embeddings']:
        init_projection= tf.keras.initializers.Constant(params['word_embeddings'].T)
    else:
        init_projection= 'glorot_uniform'
    vocab_softmax_layer= tf.keras.layers.Dense(embedding_layer.input_dim, activation=None, use_bias=True,
                                         kernel_initializer=init_projection)
    layers_dict['vocab_softmax_layer']= vocab_softmax_layer
    
    encoder_projection_layer= tf.keras.layers.Dense(params['config']['hidden_size'], activation=None, 
                                         use_bias=True, kernel_initializer='glorot_uniform')
    layers_dict['encoder_projection_layer']= encoder_projection_layer
    
    if params['pretrained_encoder'] == True:
        encoder= PretrainedEncoder(embedding_layer, encoder_projection_layer, params)
    else:
        # First create an encoder cell and then pass to encoder_fn
        enc_cell= construct_cells(params['config'], get_cell(), bidirectional=True)
        encoder= SeqEncoder(embedding_layer, encoder_projection_layer, params['config'], enc_cell)
    
    layers_dict['encoder']= encoder
    dec_cell= construct_cells(params['config'], get_cell(), bidirectional=False)['fwd']
    
    layers_dict['dec_cell']= dec_cell
    layers_dict['global_step'] = tf.train.get_global_step()
    layers_dict['temperature'] = tf.train.exponential_decay(tf.convert_to_tensor(params['init_temperature'], dtype=tf.float32),
                                            layers_dict['global_step'], 100, 0.96, staircase=True)
    
    return layers_dict


def summarization_model(features, mode, params):
    #pdb.set_trace()
    layers_dict= build_layers(features, mode, params)
    ae_encoder_output, ae_decoder_output= seq2seq_ae(features, mode, params, layers_dict)
    features['ae_encoder_output']= ae_encoder_output
    features['ae_decoder_output']= ae_decoder_output
    summ_encoder_output, summar_id_list= summarizer(features, mode, params, layers_dict)
    output_dict= {'ae_encoder_output': ae_encoder_output,
                  'ae_decoder_output': ae_decoder_output,
                  'summ_encoder_output': summ_encoder_output,
                  'summar_id_list': summar_id_list,
                  'layers_dict': layers_dict,}
    return output_dict


