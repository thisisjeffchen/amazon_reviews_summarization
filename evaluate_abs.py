import logging
logger = logging.getLogger("Training")
logger.setLevel(logging.INFO)
logging.basicConfig(format='%(levelname)s %(asctime)s : %(message)s', level=logging.INFO)
import ipdb as pdb
from ipdb import slaunch_ipdb_on_exception
import pandas as pd
import numpy as np
from rouge import Rouge
from text_encoders import NNLM, Word2Vec, USE, ELMO
from sklearn.metrics.pairwise import cosine_similarity
import ast


class MyRouge(object):
    def __init__(self):
        self.rouge= Rouge()

    def __call__(self, i, summary_list, reviewTexts):
        logging.info(i)
        summariesConcat= ". ".join (summary_list)
        total = 0
        skipped = 0
        for review in reviewTexts:
            if len(review) > 0:
                total += self.rouge.get_scores(summariesConcat, review)[0]["rouge-1"]["f"]
            else:
                skipped += 1
        rougeAvg = total / (len(reviewTexts) - skipped)
        return rougeAvg


class SemanticEval(object):
    def __init__(self):
        self.model= NNLM(trainable= False)
    
    def __call__(self, i, summary_list, reviewTexts):
        logging.info(i)
        # pdb.set_trace()
        product_embs= self.model(reviewTexts)
        summary_embs= self.model(summary_list)
        return np.mean(cosine_similarity(product_embs, summary_embs)[:,0])

    def sematic_similarity(self, product_embs, summary_embs):
        product_mean= product_embs.mean(axis= 0, keepdims=True)
        summary_mean= summary_embs.mean(axis= 0, keepdims=True)
        cosine= cosine_similarity(product_mean, summary_mean)[0][0]
        return cosine


def evaluate_rouge(input_file):
    df= pd.read_csv(input_file, encoding='latin1')
    df['input_words_list']= df['input_words_list'].apply(lambda x: ast.literal_eval(x))
    df['range']= range(len(df))
    rouge_module= MyRouge()
    df['rouge_scores']= df.apply(lambda x: rouge_module(x['range'], [x['summary']], x['input_words_list']), axis=1)
    logging.info(df['rouge_scores'].describe())

# INFO 2018-12-12 23:20:30,018 : count    500.000000
# mean       0.300872
# std        0.036273
# min        0.105621
# 25%        0.279548
# 50%        0.302436
# 75%        0.327767
# max        0.415011
# Name: rouge_scores, dtype: float64

def evaluate_semantic(input_file):
    df= pd.read_csv(input_file, encoding='latin1')
    df['input_words_list']= df['input_words_list'].apply(lambda x: ast.literal_eval(x))
    df['range']= range(len(df))
    semantic_module= SemanticEval()
    df['cosine_scores']= df.apply(lambda x: semantic_module(x['range'], [x['summary']], x['input_words_list']), axis=1)
    logging.info(df['cosine_scores'].describe())


# INFO 2018-12-12 23:57:19,707 : count    5.000000e+02
# mean     7.378698e+17
# std      1.547557e+19
# min     -3.689349e+19
# 25%     -1.084202e-19
# 50%      0.000000e+00
# 75%      2.000000e+00
# max      3.689349e+19
# Name: cosine_scores, dtype: float64

if __name__ == "__main__":
    input_file= 'results/abstractive_summaries_500_1prod_train.csv'
    # evaluate_rouge(input_file)
    evaluate_semantic(input_file)