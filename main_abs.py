import os, shutil
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

from model_data import train_input_fn, prepare_df
from modules import my_model
from config import args
os.environ['CUDA_VISIBLE_DEVICES']= "0"

def make_config():
    with open('cache/tokenizer.pkl', 'rb') as fi:
        tokenizer= pickle.load(fi)
    word_emb_size= 300
    params= {}
    params['tokenizer']= tokenizer
    params['token2id']= tokenizer.word_index
    params['vocab_size']= tokenizer.num_words
    if args.use_pretrained_embeddings:
        params['word_embeddings']= np.load('cache/pretrained_embeddings.npy')
    params['word_embeddings_dim']= word_emb_size
    params['encoder_output_size']= 512
    params['pretrained_encoder']= False
    params['learning_rate']= args.learning_rate
    params['tie_in_out_embeddings']= args.tie_in_out_embeddings
    params['init_temperature']= 2
    params['abs_num_reviews']= args.abs_num_reviews
    
    config= {}
    config['num_layers']= 1
    config['hidden_size']= 512
    config['dropout_keep']= 0.9
    params['config']= config
    return params
    

def id_to_text(tokenizer, id_list):
    max_lens= [len(seq) if 0 not in seq else seq.index(0) for seq in id_list]
    words_list= tokenizer.sequences_to_texts([ids[:max_lens[i]] for i, ids in enumerate(id_list)])
    return words_list


def train_model(classifier, params, train_filename):
    features_df, word_ids= prepare_df(asins2use_file= train_filename)
    # Train the Model.
    if args.debug == True:
        maxsteps= None
    else:
        maxsteps= None
    classifier.train(
        input_fn=lambda: train_input_fn(features_df, word_ids),
        steps=maxsteps)


def test_model(classifier, params, test_filename):
    features_df, word_ids= prepare_df(asins2use_file= test_filename)
    # Test the Model.
    predictions = classifier.predict(input_fn=lambda: train_input_fn(features_df, word_ids))
    asin_list, summary_id_list=[], []
    ae_ids_list, input_ids_list= [], []
    for i, pred_dict in enumerate(predictions):
        if i==0:
            pdb.set_trace()
        print ("Processing {}".format(i))
        if args.debug == True and i > 500:
            break
        elif i == 500:
            break
        asin_list.append(pred_dict['asin'][0].decode())
        summary_id_list.append(pred_dict['summary_ids'].tolist())
        ae_ids_list.append(pred_dict['ae_word_ids'].tolist())
        input_ids_list.append(pred_dict['input_word_ids'].tolist())

    pdb.set_trace()
    tokenizer= params['tokenizer']
    summary_words_list= id_to_text(tokenizer, summary_id_list)
    ae_words_list= [id_to_text(tokenizer, word_ids) for word_ids in ae_ids_list]
    input_words_list= [id_to_text(tokenizer, word_ids) for word_ids in input_ids_list]
    
    ddict= defaultdict(list)
    out_dict= OrderedDict()
    for i, summary in enumerate(summary_words_list):
        asin= asin_list[i]
        out_dict[asin]= {}
        out_dict[asin]['summary']= summary
        out_dict[asin]['ae_words_list']= ae_words_list[i]
        out_dict[asin]['input_words_list']= input_words_list[i]
        ddict['asin'].append(asin_list[i])
        ddict['summary'].append(summary)
        ddict['ae_words_list'].append(ae_words_list[i])
        ddict['input_words_list'].append(input_words_list[i])
    
    with open('results/abstractive_summaries.json', 'w') as fo:
        json.dump(out_dict, fo, ensure_ascii=False, indent=2)
    
    df= pd.DataFrame(ddict)
    df.to_csv('results/abstractive_summaries.csv')
    

def safe_mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def run_model():
    pdb.set_trace()
    params= make_config()
    num_reviews= args.abs_num_reviews
    train_filename, test_filename= 'abs_train_set_{}.csv'.format(num_reviews), 'abs_test_set_{}.csv'.format(num_reviews)
    model_dir= 'cache/checkpoints'
    if args.cold_start:
        shutil.rmtree(model_dir, ignore_errors=True)
        os.makedirs(model_dir, exist_ok=True)
    else:
        safe_mkdir(model_dir)
    model_config= tf.estimator.RunConfig(model_dir=model_dir,
                                        tf_random_seed=42,
                                        log_step_count_steps=10,
                                        save_checkpoints_steps=50,
                                        keep_checkpoint_max=3)
    classifier = tf.estimator.Estimator(
        model_fn= my_model,
        params= params,
        config= model_config)

    # train_model(classifier, params, train_filename)
    pdb.set_trace()
    # test_model(classifier, params, train_filename)
    test_model(classifier, params, test_filename)


if args.debug == False:
    pdb.set_trace= lambda:None


if __name__ == "__main__":
    with slaunch_ipdb_on_exception():
        run_model()

