#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 14:47:08 2018

@author: hanozbhathena
"""

import os
import gzip
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import shelve
import sqlite3
import logging
logger = logging.getLogger("my_logger")
logger.setLevel(logging.INFO)
logging.basicConfig(format='%(levelname)s %(asctime)s : %(message)s', level=logging.INFO)
import ipdb as pdb
from ipdb import slaunch_ipdb_on_exception
from collections import defaultdict
import json
import ast
import time


def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)


def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
        if i % 1000 == 0:
            print(i)
            break
#            df= pd.DataFrame.from_dict(df, orient='index')
#            df= {}
    return pd.DataFrame.from_dict(df, orient='index')


def test_repeats(path):
    unique_asins= set()
    asin_counts= defaultdict(int)
    for i, dd in enumerate(parse(path)):
        curr_asin= dd['asin']
        asin_counts[curr_asin]+=1
        if curr_asin in unique_asins:
            raise ValueError("Not in order")
        if i==0:
            prev_asin= curr_asin
            continue
        if prev_asin != curr_asin:
            unique_asins.add(prev_asin)
        prev_asin= curr_asin
        if i>0 and i%100000==0:
            print(i)
    print("Done")
    return asin_counts,unique_asins


def test_shelve():
    d = shelve.open("debug.shelf")
    for i in range(0, 1000000):
        d[str(i)] = str(i*2)
    d.close()


def insert(cur, asin, reviewText, overall):
    cur.execute("INSERT INTO reviews_dict (asin, reviewText, overall) VALUES (?, ?, ?)",
                (str(asin), str(reviewText), str(overall)))


def create_review_db(data_dir, raw_review_file):
    db_file= os.path.join(data_dir, "reviews.s3db")
    conn= sqlite3.connect(db_file)
    cur= conn.cursor()
    cur.execute("drop table if exists reviews_dict;")
    cur.execute("CREATE TABLE IF NOT EXISTS reviews_dict ("
                "asin VARCHAR(255) PRIMARY KEY NOT NULL, "
                "reviewText VARCHAR(255),"
                "overall VARCHAR(255))")
    
    unique_asins= set()
    asin_counts= defaultdict(int)
    prev_asin= None
    text_list, rating_list= [], []
    #try:
    for i, dd in enumerate(parse(raw_review_file)):
        curr_asin= dd['asin']
        asin_counts[curr_asin]+=1
        if curr_asin in unique_asins:
            raise ValueError("Not in order")
        if prev_asin != curr_asin:
            unique_asins.add(prev_asin)
            insert(cur, prev_asin, text_list, rating_list)
            text_list, rating_list= [], []
        prev_asin= curr_asin
        text_list.append(dd['reviewText'])
        rating_list.append(dd['overall'])
        if i>0 and i%100000==0:
            logging.info("Done: {}".format(i))
        if i>0 and i%1000000==0:
            conn.commit()
#            if i>300000:
#                break
    insert(cur, prev_asin, text_list, rating_list)
    conn.commit()
    df= pd.DataFrame.from_dict(asin_counts, orient='index')
    print(df.describe())
    df['asin']= df.index
    df.columns= ['num_reviews', 'asin']
    df.to_csv('num_reviews.csv', index=False)
    # except Exception as e:
    #     logging.info("Error type: {}".format(type(e).__name__))
    #     conn.rollback()
    #finally:
    cur.close()
    conn.close()


class SQLLiteIndexer(object):
    def __init__(self, data_dir, attribute= "reviewText", table_name= "reviews_dict", 
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
    
    def __del__(self):
        print("Closing SQLLite connection")
        self.cur.close()
        self.conn.close()


class SQLLiteBatchIterator(object):
    def __init__(self, asin_df_fname, data_dir, attribute= "reviewText", table_name= "reviews_dict", 
                 db_file= "reviews.s3db", asin_chunksize= 1000, num_chunks= None, offset= 0):
#        pdb.set_trace()
        db_file= os.path.join(data_dir, db_file)
        self.conn= sqlite3.connect(db_file)
        self.cur= self.conn.cursor()
        self.attribute= attribute
        self.table_name= table_name
        self.asin_df_fname= asin_df_fname
        self.asin_chunksize= asin_chunksize
        self.num_chunks= num_chunks
        self.skip_row_list= list(range(1, 1 + offset))

    
    def __iter__(self):
        self.dfi= pd.read_csv(self.asin_df_fname, encoding= 'latin1', chunksize= self.asin_chunksize,
                          nrows= self.asin_chunksize * self.num_chunks, skiprows= self.skip_row_list)
        for i, df_chunk in enumerate(self.dfi):
            asin_list= df_chunk.asin.tolist()
            self.cur.execute("""
                     SELECT {attribute} from {table_name} 
                     where asin in ({num_qs})
                     """.format(attribute= self.attribute, 
                     table_name= self.table_name, num_qs=','.join('?'*len(asin_list))), asin_list)
            temp_list= self.cur.fetchall()
            temp_list= [ast.literal_eval(temp[0]) for temp in temp_list]
            yield temp_list
    
    def __del__(self):
        print("Closing SQLLite connection")
        self.cur.close()
        self.conn.close()

