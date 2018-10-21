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
  embedded = embed (all_sentences)  

  print (embedded)

  print (len(all_sentences))
  print (len(embedded))


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
    text = review [reviewText]
    sentences = text.split (". ")
    for sent in sentences:
      all_sentences.append (sent)
      review_indicies.append (idx)

  return all_sentences, review_indicies

def embed (all_sentences):
  embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/3")
  embeddings = embed(all_sentences)
  return embeddings

def cluster ():
  pass

if __name__ == '__main__':
    main()