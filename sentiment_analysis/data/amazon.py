"""Amazon Dataset module for sentiment analysis."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import pdb
import os
import sqlite3
import ast

from data.util import OOV_CHAR
from data.util import pad_sentence
from data.util import START_CHAR
from keras.preprocessing.text import Tokenizer

import pickle

#import sys
#sys.path.append("..")
#from data_utils import SQLLiteIndexer

NUM_CLASS = 3
DATA_PATH= os.environ.get('DATA_PATH') or './data/'


  #TODO: ugly, but why doesn't python let me import from data utils in above directory?
class SQLLiteIndexer(object):
  #TODO: ugly
  def __init__(self, data_dir = DATA_PATH, attribute= "reviewText", table_name= "reviews_dict", 
               db_file= "reviews.s3db"):
      db_file= os.path.join(data_dir, db_file)
      self.conn= sqlite3.connect(db_file)
      self.cur= self.conn.cursor()
      self.attribute= attribute
      self.table_name= table_name
  
  def __getitem__(self, asin):
      self.cur.execute("""
               SELECT {attribute} from {table_name} 
               where asin=?
               """.format(attribute= self.attribute, 
               table_name= self.table_name), (asin,))
      return ast.literal_eval(self.cur.fetchall()[0][0])
      #TODO TURN OFF DEBUG MODE FOR LIMIT 100
  def getall(self, attributes = "reviewShort, ratingShort"):
      self.cur.execute ("""
              SELECT {attributes} from {table_name}
              LIMIT 100000
              """.format (attributes = attributes, 
                          table_name = self.table_name))
      fetched = self.cur.fetchall()
      fetched_list = [(ast.literal_eval(row[0]), ast.literal_eval(row[1])) for row in fetched]
      return fetched_list
  
  def __del__(self):
      print("Closing SQLLite connection")
      self.cur.close()
      self.conn.close()

def save_tokenizer (tokenizer):
  with open('cache/tokenizer_sentiment.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_tokenizer ():
  with open('cache/tokenizer_sentiment.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def load(vocabulary_size, sentence_length, num_samples):
  """Returns training and evaluation input for imdb dataset.

  Args:
    vocabulary_size: The number of the most frequent tokens
      to be used from the corpus.
    sentence_length: The number of words in each sentence.
      Longer sentences get cut, shorter ones padded.
  Raises:
    ValueError: if the dataset value is not valid.
  Returns:
    A tuple of length 4, for training and evaluation data,
    each being an numpy array.
  """
  print ("Fetching data from DB...")
  db = SQLLiteIndexer(db_file = "reviews.s3db")
  rows = db.getall()
  x = []
  y = []

  print ("Adding rows to train/test sets")
  for row in rows:
    assert len(row[0]) == len(row[1])
    for idx in range (len(row[0])):
      x.append(row[0][idx])
      y.append(row[1][idx])

  #convert y to integers
  y = [int(elem) for elem in y]

  print ("Total reviews: {}, currently only using {}".format(len(x), num_samples))
  #Currently only using 100k

  if num_samples == 100000:
    x_train = x[0:80000]
    y_train = y[0:80000]
    x_test = x[80001:100000]
    y_test = y[80001:100000]
  elif num_samples == 1000000:
    x_train = x[0:800000]
    y_train = y[0:800000]
    x_test = x[800001:1000000]
    y_test = y[800001:1000000]


  print ("Fitting tokenizer...")
  #TODO: not using start and end chars, does it matter?
  tokenizer = Tokenizer (num_words = vocabulary_size, oov_token = OOV_CHAR)
  tokenizer.fit_on_texts(x_train)
  print ("Tokenizing...")
  x_train = tokenizer.texts_to_sequences(x_train)
  x_test = tokenizer.texts_to_sequences(x_test)

  #pdb.set_trace ()

  print ("Padding...")
  x_train_processed = []
  for sen in x_train:
    sen = pad_sentence(sen, sentence_length)
    x_train_processed.append(np.array(sen))
  x_train_processed = np.array(x_train_processed)


  x_test_processed = []
  for sen in x_test:
    sen = pad_sentence(sen, sentence_length)
    x_test_processed.append(np.array(sen))
  x_test_processed = np.array(x_test_processed)

  print ("Saving tokenizer...")
  save_tokenizer (tokenizer)

  #pdb.set_trace()

  return x_train_processed, np.eye(NUM_CLASS)[y_train], \
         x_test_processed, np.eye(NUM_CLASS)[y_test]


