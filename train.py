import argparse
import json
import pdb
import tensorflow as tf
import tensorflow_hub as hub

REVIEWS_PATH = "data/processed/single_product.json"

def main():
  parser = argparse.ArgumentParser ()
  parser.add_argument ('--embed', type=str, default="w2v",
                       help="choose amongst different models")

  reviews = get_reviews ()
  all_sentences, review_indicies = segment (reviews)
  print (all_sentences)
  all_sentences=all_sentences[0:10]
  embedded = embed (all_sentences)  

  
  print (len(all_sentences))
  print (len(embedded))

  print (embedded)
  #grab a model

def get_reviews ():
  print ("here")
  with open(REVIEWS_PATH) as f:
    data = json.load(f)
  return data

def segment (reviews):
  all_sentences = []
  review_indicies = []
  for idx in range (len (reviews)):
    review = reviews[idx]
    text = review ['reviewText']
    sentences = text.split (". ")
    for sent in sentences:
      all_sentences.append (sent)
      review_indicies.append (idx)

  return all_sentences, review_indicies

def embed (all_sentences):
  with tf.Graph().as_default():
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