"""Dataset module for sentiment analysis.

Currently imdb dataset is available.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import data.imdb as imdb
import data.amazon as amazon

DATASET_IMDB = "imdb"
DATASET_AMAZON = "amazon"


def load(dataset, vocabulary_size, sentence_length, num_samples):
  """Returns training and evaluation input.

  Args:
    dataset: Dataset to be trained and evaluated.
      Currently only imdb is supported.
    vocabulary_size: The number of the most frequent tokens
      to be used from the corpus.
    sentence_length: The number of words in each sentence.
      Longer sentences get cut, shorter ones padded.
  Raises:
    ValueError: if the dataset value is not valid.
  Returns:
    A tuple of length 4, for training sentences, labels,
    evaluation sentences, and evaluation labels,
    each being an numpy array.
  """

  if dataset == DATASET_IMDB:
    return imdb.load(vocabulary_size, sentence_length)
  elif dataset == DATASET_AMAZON:
    return amazon.load(vocabulary_size, sentence_length, num_samples)
  else:
    raise ValueError("unsupported dataset: " + dataset)


def get_num_class(dataset):
  """Returns an integer for the number of label classes.

  Args:
    dataset: Dataset to be trained and evaluated.
      Currently only imdb is supported.
  Raises:
    ValueError: if the dataset value is not valid.
  Returns:
    int: The number of label classes.
  """
  if dataset == DATASET_IMDB:
    return imdb.NUM_CLASS
  elif dataset == DATASET_AMAZON:
    return amazon.NUM_CLASS
  else:
    raise ValueError("unsupported dataset: " + dataset)
