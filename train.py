import numpy as np
import argparse
import json
import pdb
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.cluster import KMeans
from nltk.tokenize import sent_tokenize


REVIEWS_PATH = "data/processed/"
PICKED_PRODUCTS = ["B00001WRSJ", "B009AYLDSU", "B007I5JT4S"]
RESULTS_PATH = "results/"

def main():
  parser = argparse.ArgumentParser ()
  parser.add_argument ('--embed', type=str, default="nnlm",
                       help="choose amongst different models")
  parser.add_argument ('--clusters', type=int, default=5,
                       help="5 clusters for kmeans")
  args = parser.parse_args ()

  for product in PICKED_PRODUCTS:
    reviews = get_reviews (args, product)
    #all_sentences, review_indicies = segment (args, reviews)
    #print (all_sentences)
    #all_sentences=all_sentences[0:10]
  
    reviewTexts = [review['reviewText'] for review in reviews ]
    all_sentences = [sent for review in reviewTexts for sent in sent_tokenize(review)]
  
    print ("Embedding...")
    embedded = embed (args, all_sentences)  
  
    print ("Running kmeans...")
    clusters = KMeans(n_clusters = args.clusters, random_state = 0).fit (embedded)  
    for centroid in range(1, args.clusters + 1):
      

    dist = clusters.transform (embedded)
    product_reviews_np = np.array (all_sentences)
    summaries = product_reviews_np[np.argmin (dist, axis = 0)]
  
    with open(RESULTS_PATH + "5_summary_" + product + ".json", 'w') as f:
      json.dump(summaries.tolist(), f, ensure_ascii=False, indent=2)


def get_reviews (args, product):
  with open(REVIEWS_PATH + product + ".json") as f:
    data = json.load(f)
  return data

# def segment (args, reviews):
#   all_sentences = []
#   review_indicies = []
#   for idx in range (len (reviews)):
#     review = reviews[idx]
#     text = review ['reviewText']
#     sentences = text.split (". ")
#     for sent in sentences:
#       all_sentences.append (sent)
#       review_indicies.append (idx)

#   return all_sentences, review_indicies

def embed (args, all_sentences):
  with tf.Graph().as_default():
    if args.embed == "nnlm":
      module_url = "https://tfhub.dev/google/nnlm-en-dim128-with-normalization/1"
      embed = hub.Module(module_url)
      embeddings = embed(all_sentences)

      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        return sess.run(embeddings)

def cluster ():
  pass

if __name__ == '__main__':
    main()