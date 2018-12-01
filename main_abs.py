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
from modules import my_model
from config import args


def make_config():
    with open('cache/tokenizer.pkl', 'rb') as fi:
        tokenizer= pickle.load(fi)
    word_emb_size= 300
    params= {}
    params['tokenizer']= tokenizer
    params['token2id']= tokenizer.word_index
    params['vocab_size']= tokenizer.num_words
    params['word_embeddings']= np.load('cache/pretrained_embeddings.npy')
    params['word_embeddings_dim']= word_emb_size
    params['encoder_output_size']= 512
    params['pretrained_encoder']= False
    params['learning_rate']= args.learning_rate
    params['tie_in_out_embeddings']= args.tie_in_out_embeddings
    
    config= {}
    config['num_layers']= 1
    config['hidden_size']= 512
    config['dropout_keep']= 0.9
    params['config']= config
    return params
    

def train_model(classifier, params, train_filename):
    # Train the Model.
    if args.debug == True:
        maxsteps= 50
    else:
        maxsteps= None
    classifier.train(
        input_fn=lambda: train_input_fn(asins2use_file= train_filename),
        steps=maxsteps)


def test_model(classifier, params, test_filename):
    # Test the Model.
    predictions = classifier.predict(input_fn=lambda: train_input_fn(asins2use_file= test_filename))
    asin_list, summary_id_list=[], []
    for i, pred_dict in enumerate(predictions):
        # pdb.set_trace()
        print ("Processing {}".format(i))
        if args.debug == True and i > 10:
            break
        elif i == 100:
            break
        asin_list.append(pred_dict['asin'])
        summary_id_list.append(pred_dict['summary_ids'].tolist())

    pdb.set_trace()
    tokenizer= params['tokenizer']
    max_lens= [len(seq) if 0 not in seq else seq.index(0) for seq in summary_id_list]
    summary_words_list= tokenizer.sequences_to_texts([ids[:max_lens[i]] for i, ids in enumerate(summary_id_list)])
    
    # out_dict= defaultdict(list)
    out_dict= OrderedDict()
    for i, summary in enumerate(summary_words_list):
        asin= asin_list[i].decode()
        out_dict[asin]= {}
        out_dict[asin]['summary']= summary
        # out_dict['asin'].append(asin_list[i].decode())
        # out_dict['summary'].append(summary)
    
    with open('results/abstractive_summaries.json', 'w') as fo:
        json.dump(out_dict, fo, ensure_ascii=False, indent=2)
    

def safe_mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def run_model():
    # pdb.set_trace()
    params= make_config()
    num_reviews= args.abs_num_reviews
    train_filename, test_filename= 'abs_train_set_{}.csv'.format(num_reviews), 'abs_test_set_{}.csv'.format(num_reviews)
    model_dir= '../abs_model_dir'
    if args.cold_start:
        os.makedirs(model_dir, exist_ok=True)
    else:
        safe_mkdir(model_dir)
    model_config= tf.estimator.RunConfig(model_dir=model_dir,
                                        tf_random_seed=42,
                                        log_step_count_steps=10,
                                        save_checkpoints_steps=100,
                                        keep_checkpoint_max=3)
    classifier = tf.estimator.Estimator(
        model_fn= my_model,
        params= params,
        config= model_config)
    

   # train_model(classifier, params, train_filename)
    pdb.set_trace()
    test_model(classifier, params, test_filename)


if args.debug == False:
    pdb.set_trace= lambda:None


if __name__ == "__main__":
    with slaunch_ipdb_on_exception():
        run_model()

