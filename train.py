import numpy as np
import argparse
import json
import pdb
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.cluster import KMeans
from nltk.tokenize import sent_tokenize
from rouge import Rouge


REVIEWS_PATH = "data/processed/"
PICKED_PRODUCTS = ["B000EI0EB8", "B0007OWASE", "B00008OE43"]
#PICKED_PRODUCTS = ["B00001WRSJ", "B009AYLDSU", "B007I5JT4S", "B000EI0EB8", "B0007OWASE", "B00008OE43"]
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
    sentenceParent = []
    all_sentences = []
    for idx, review in enumerate(reviewTexts):
      for sent in sent_tokenize(review):
        all_sentences.append(sent)
        sentenceParent.append(idx)

    counts = []
  
    print ("Embedding...")
    embedded = embed (args, all_sentences)  
  
    print ("Running kmeans...")
    clusters = KMeans(n_clusters = args.clusters, random_state = 0).fit (embedded)  
    for label in range(args.clusters):
      reviewsForLabel = []
      for idx, reviewLabel in enumerate(clusters.labels_):
        if reviewLabel == label:
          reviewsForLabel.append (sentenceParent[idx])
      count = len ( set (reviewsForLabel))
      counts.append (count)      

    dist = clusters.transform (embedded)
    product_reviews_np = np.array (all_sentences)
    summaries = product_reviews_np[np.argmin (dist, axis = 0)].tolist()

    #compute rouge1
    summariesConcat = ". ".join (summaries)
    total = 0
    rouge = Rouge()

    for review in reviewTexts:
      total += rouge.get_scores (summariesConcat, review)[0]["rouge-1"]["f"]

    rougeAvg = total / len(reviewTexts)

    bestFitAverages = []
    #compute best-fit rouge1
    for idx, sentence in enumerate (all_sentences):
      sentReviewTotal = 0
      for review in reviewTexts:
        sentReviewTotal += rouge.get_scores (sentence, review)[0]["rouge-1"]["f"]
      bestFitAverages.append((sentReviewTotal / len(reviewTexts), idx))

    bestFitSorted = sorted(bestFitAverages, reverse = True)

    bestFitSummaries = [all_sentences[element[1]] for element in bestFitSorted[0:args.clusters]]

    bestFitSummariesConcat = ". ".join (bestFitSummaries)
    bestFitTotal = 0

    #ugly
    for review in reviewTexts:
      bestFitTotal += rouge.get_scores (bestFitSummariesConcat, review)[0]["rouge-1"]["f"]

    rougeBestFitAvg = bestFitTotal / len(reviewTexts)

  
    with open(RESULTS_PATH + "5_summary_" + product + ".json", 'w') as f:
      json.dump(summaries, f, ensure_ascii=False, indent=2)
      json.dump(counts, f, ensure_ascii=False, indent=2)
      json.dump(rougeAvg, f, ensure_ascii=False, indent=2)
      json.dump(bestFitSummaries, f, ensure_ascii=False, indent=2)
      json.dump(rougeBestFitAvg, f, ensure_ascii=False, indent=2)


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