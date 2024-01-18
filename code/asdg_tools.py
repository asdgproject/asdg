#!/usr/bin/env python


import os,sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import pickle
from bertopic import BERTopic

from collections import Counter

import gensim
from gensim.parsing.preprocessing import STOPWORDS

from sentence_transformers import SentenceTransformer

from nltk.stem import WordNetLemmatizer

def lemmatize_stemming(text):
    '''
    Function to lemmatize. NOT USED
    '''
    lemmatizer = WordNetLemmatizer()
    return lemmatizer.lemmatize(text)

def preprocess(text, lemma=False, ext_stopwords_flag=False, ext_stopwords=[]):
    '''
    Function to preprocess texts before being used as input to train models.
    Input:
        - text: list of sentences.
        - lemma: lemmatizer flag, if set to True lemmatization is performed on every word.
        - ext_stopwords_flag: externalstop words usage flag, if set to True these external stopwords are used.
        - ext_stopwords: external stopwords list.
    Output:
        - result: list of preprocessed tokenized sentences.
    '''
    result = []
    if lemma: lemmatizer = WordNetLemmatizer()
    for token in gensim.utils.simple_preprocess(text):
        if ext_stopwords:
            if token not in gensim.parsing.preprocessing.STOPWORDS and token not in ext_stopwords and len(token) > 3:
                if lemma:
                    result.append(lemmatizer.lemmatize(token))
                else:
                    result.append(token)
        else:
            if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
                if lemma:
                    result.append(lemmatizer.lemmatize(token))
                else:
                    result.append(token)
    return result

def load_custom_stopwords(sw_file, sep=" "):
    '''
    Function to load custom stopwords from a file.
    Separator = " ". Single line file.
    Input:
        - path to the file containing the stopwords.
    Output:
        - list of stopwords.
    '''
    with open(sw_file) as f:
        lines = f.readlines()
        lines[0] = lines[0].strip()
        ext_stopwords = lines[0].split(sep)
    return ext_stopwords

def load_training_data(files_path):
    '''
    Function to load training data from parts files (txt files).
    Input:
        - files_path: path to the folder containing the files to train a model.
    Output:
        - SDG_df: pandas DataFrame with "text" column including the text in parts files.
    '''
    files_SDG_parts = [x[:-4] for x in os.listdir(files_path) if x.endswith(".txt")]
    files_SDG_parts.sort()
    SDG_df = pd.DataFrame(columns=["SDG", "text"])
    SDG_df["SDG"] = files_SDG_parts
    for i in range(len(files_SDG_parts)):
        f = open(files_path+files_SDG_parts[i]+".txt", 'r')
        text = f.read()
        f.close()
        SDG_df.at[i,"text"] = text
    return SDG_df

def train_BERTopic(train_df, n_words, n_topics, ext_stopwords=[]):
    '''
    Function to train BERTopic.
    Input:
        - train_df: pandas DataFrame containing the text to train the model. Text is assumed to be in the column "text".
        - n_words: top_n_words to be displayed
        - n_topics: number of topics
    Output:
        - model: BERTopic model trained for parts' text
        - topics: topics extracted by BERTopic.
        - probs: probabilities for each topic.
    '''
    sentence_model = SentenceTransformer("all-mpnet-base-v2")
    processed_docs_SDG = train_df['text'].apply(lambda x: preprocess(x, lemma=True, ext_stopwords_flag=True, ext_stopwords=ext_stopwords))
    processed_docs_SDG = processed_docs_SDG.tolist()

    clean_docs = []
    for i in range(len(processed_docs_SDG)):
        aux = " ".join(processed_docs_SDG[i])
        clean_docs.append(aux)
        
    model = BERTopic(language="english", verbose=True, top_n_words=10, nr_topics=17, embedding_model=sentence_model, min_topic_size=3, calculate_probabilities=True)
    
    topics, probs = model.fit_transform(clean_docs) 
    
    return model, topics, probs

def load_BERTopic_model(model):
    '''
    Function to load a pretrained BERTopic model.
    Input:
        - path to the model
    Output:
        - model
    '''
    return BERTopic.load(model)
    
def load_topic2SDG_map(path):
    '''
    Function to load the topic to SDG mapping.
    Input:
        - path to the saved dictionary containing the mapping.
    Output:
        - topic to SDG dictionary
    '''
    with open(path, 'rb') as handle:
        return pickle.load(handle)





