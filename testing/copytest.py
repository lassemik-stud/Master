import os
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator, CEBinaryClassificationEvaluator
from torch.utils.data import DataLoader
import math
import multiprocessing as multiprocessing
from sentence_transformers import LoggingHandler, util
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
from sentence_transformers import InputExample
import logging
import json
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc
import torch
import torch.nn as nn
from torch.utils import data
import copy
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from transformers import BertTokenizerFast
import datetime
import time
import nltk
nltk.download('punkt')


import numpy as np
import re
import string
import os.path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
import nltk
from nltk.corpus import stopwords
from itertools import chain
from nltk.tag.perceptron import PerceptronTagger
from nltk.corpus import conll2000
import pickle
import nltk
import nltk.data





chunker_instance = None
tokenizer = nltk.tokenize.TreebankWordTokenizer()
tagger = nltk.download('treebank')

grammar = r"""
  NP: 
      {<DT|WDT|PP\$|PRP\$>?<\#|CD>*(<JJ|JJS|JJR><VBG|VBN>?)*(<NN|NNS|NNP|NNPS>(<''><POS>)?)+}
      {<DT|WDT|PP\$|PRP\$><JJ|JJS|JJR>*<CD>}
      {<DT|WDT|PP\$|PRP\$>?<CD>?(<JJ|JJS|JJR><VBG>?)}
      {<DT>?<PRP|PRP\$>}
      {<WP|WP\$>}
      {<DT|WDT>}
      {<JJR>}
      {<EX>}
      {<CD>+}
  VP: {<VBZ><VBG>}
      {(<MD|TO|RB.*|VB|VBD|VBN|VBP|VBZ>)+}
      

"""

def get_nltk_pos_tag_based_chunker():
    global chunker_instance
    if chunker_instance is not None:
        return chunker_instance
    chunker_instance = nltk.RegexpParser(grammar)
    return chunker_instance

    

def chunk_to_str(chunk):
    if type(chunk) is nltk.tree.Tree:
        return chunk.label()
    else:
        return chunk[1]

def extract_subtree_expansions(t, res):
    if type(t) is nltk.tree.Tree:
        expansion = t.label() + "[" + " ".join([chunk_to_str(child) for child in t]) + "]"
        res.append(expansion)
        for child in t:
            extract_subtree_expansions(child, res)
            
def nltk_pos_tag_chunk(pos_tags):
    chunker = get_nltk_pos_tag_based_chunker()
    parse_tree = chunker.parse(pos_tags)
    subtree_expansions = []
    for subt in parse_tree:
        extract_subtree_expansions(subt, subtree_expansions)
    return list(map(chunk_to_str, parse_tree)), subtree_expansions


def prepare_entry(text):
    tokens = []
    # Workaround because there re some docuemtns that are repitions of the same word which causes the regex chunker to hang
    prev_token = ''
    for t in tokenizer.tokenize(text):
        if t != prev_token:
            tokens.append(t)
    tagger_output = nltk.pos_tag(tokens)
    pos_tags = [t[1] for t in tagger_output]
    pos_chunks, subtree_expansions = nltk_pos_tag_chunk(tagger_output)
    entry = {
        'preprocessed': text,
        'pos_tags': pos_tags,
        'pos_tag_chunks': pos_chunks,
        'pos_tag_chunk_subtrees': subtree_expansions,
        'tokens': tokens
    }
    return entry



def word_count(entry):
    return len(entry['tokens'])

def avg_chars_per_word(entry):
    r = np.mean([len(t) for t in entry['tokens']])
    return r

def distr_chars_per_word(entry, max_chars=10):
    counts = [0] * max_chars
    for t in entry['tokens']:
        l = len(t)
        if l <= max_chars:
            counts[l - 1] += 1
    r = [c/len(entry['tokens']) for c in counts]
#     fnames = ['distr_chars_per_word_' + str(i + 1)  for i in range(max_chars)]
    return r
    
def character_count(entry):
    r = len(re.sub('\s+', '', entry['preprocessed']))
    return r


#https://github.com/ashenoy95/writeprints-static/blob/master/whiteprints-static.py
def hapax_legomena(entry):
    freq = nltk.FreqDist(word for word in entry['tokens'])
    hapax = [key for key, val in freq.items() if val == 1]
    dis = [key for key, val in freq.items() if val == 2]
    if len(dis) == 0 or len(entry['tokens']) == 0:
        return 0
    return (len(hapax) / len(dis)) / len(entry['tokens'])

def n_of_pronouns(entry):
    # Gab: aggiunto da noi
    return entry['pos_tags'].count('PRP') + entry['pos_tags'].count('PRP$')

def gunning_fog_index(entry):
    # Gab: aggiunto da noi
    #r = Readability(entry['preprocessed'])
    gf = r.gunning_fog()
    return gf.score




class CustomTfIdfTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, key, analyzer, n=1, vocab=None):
        self.key = key
        self.pass_fn = lambda x: x
        if self.key == 'pos_tags' or self.key == 'tokens' or self.key == 'pos_tag_chunks' or self.key == 'pos_tag_chunk_subtrees':
            self.vectorizer = TfidfVectorizer(analyzer=analyzer, min_df=0.1, tokenizer=self.pass_fn, preprocessor=self.pass_fn, vocabulary=vocab, norm='l1', ngram_range=(1, n))
        else:
            self.vectorizer = TfidfVectorizer(analyzer=analyzer, min_df=0.1, vocabulary=vocab, norm='l1', ngram_range=(1, n))

    def fit(self, x, y=None):
        self.vectorizer.fit([entry[self.key] for entry in x], y)
        return self

    def transform(self, x):
        return self.vectorizer.transform([entry[self.key] for entry in x])
    
    def get_feature_names(self):
        return self.vectorizer.get_feature_names()
    
    
class CustomFreqTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, analyzer, n=1, vocab=None):
        self.pass_fn = lambda x: x
        self.vectorizer = TfidfVectorizer(tokenizer=self.pass_fn, preprocessor=self.pass_fn, vocabulary=vocab, norm=None, ngram_range=(1, n))

    def fit(self, x, y=None):
        self.vectorizer.fit([entry['tokens'] for entry in x], y)
        return self

    def transform(self, x):
        d = np.array([1 + len(entry['tokens']) for entry in x])[:, None]
        return self.vectorizer.transform([entry['tokens'] for entry in x]) / d
    
    def get_feature_names(self):
        return self.vectorizer.get_feature_names()
    
    
class CustomFuncTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, transformer_func, fnames=None):
        self.transformer_func = transformer_func
        self.fnames = fnames
        
    def fit(self, x, y=None):
        return self
    
    def transform(self, x):
        xx = np.array([self.transformer_func(entry) for entry in x])
        if len(xx.shape) == 1:
            return xx[:, None]
        else:
            return xx
    
    def get_feature_names(self):
        if self.fnames is None:
            return ['']
        else:
            return self.fnames
        
        
def get_writeprints_transformer():
    char_distr = CustomTfIdfTransformer('preprocessed', 'char_wb', n=6)
    word_distr = CustomTfIdfTransformer('preprocessed', 'word', n=3)
    pos_tag_distr = CustomTfIdfTransformer('pos_tags', 'word', n=3)
    pos_tag_chunks_distr = CustomTfIdfTransformer('pos_tag_chunks', 'word', n=3)
    pos_tag_chunks_subtree_distr = CustomTfIdfTransformer('pos_tag_chunk_subtrees', 'word', n=1)
    punctuation = '!"#$%&\'()*+,-./:;<=>?@[\]^_`{¦}~'
    special_char_distr = CustomTfIdfTransformer('preprocessed', 'char_wb', vocab=punctuation)
    freq_func_words = CustomFreqTransformer('word', vocab=stopwords.words('english'))

    transformer = FeatureUnion([
        ('char_distr', char_distr),
        ('word_distr', word_distr),
        ('pos_tag_distr', pos_tag_distr),
        ('pos_tag_chunks_distr', pos_tag_chunks_distr),
        ('pos_tag_chunks_subtree_distr', pos_tag_chunks_subtree_distr),
        ('special_char_distr', special_char_distr),
        ('freq_func_words', freq_func_words),
        ('hapax_legomena', CustomFuncTransformer(hapax_legomena)),
        ('character_count', CustomFuncTransformer(character_count)),
        ('distr_chars_per_word', CustomFuncTransformer(distr_chars_per_word, fnames=[str(i) for i in range(10)])),
        ('avg_chars_per_word', CustomFuncTransformer(avg_chars_per_word)),
        ('word_count', CustomFuncTransformer(word_count)),
        #('readability', CustomFuncTransformer(gunning_fog_index) ),
        ('n_of_pronouns', CustomFuncTransformer(n_of_pronouns) )
    ], n_jobs=-1)
    
    return transformer


import argparse
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score


def binarize(y, threshold=0.5):
    y = np.array(y)
    y = np.ma.fix_invalid(y, fill_value=threshold)
    y[y >= threshold] = 1
    y[y < threshold] = 0
    return y


def auc(true_y, pred_y):
    """
    Calculates the AUC score (Area Under the Curve), a well-known
    scalar evaluation score for binary classifiers. This score
    also considers "unanswered" problem, where score = 0.5.

    Parameters
    ----------
    prediction_scores : array [n_problems]

        The predictions outputted by a verification system.
        Assumes `0 >= prediction <=1`.

    ground_truth_scores : array [n_problems]

        The gold annotations provided for each problem.
        Will typically be `0` or `1`.

    Returns
    ----------
    auc = the Area Under the Curve.

    References
    ----------
        E. Stamatatos, et al. Overview of the Author Identification
        Task at PAN 2014. CLEF (Working Notes) 2014: 877-897.

    """
    try:
        return roc_auc_score(true_y, pred_y)
    except ValueError:
        return 0.0


def c_at_1(true_y, pred_y, threshold=0.5):
    """
    Calculates the c@1 score, an evaluation method specific to the
    PAN competition. This method rewards predictions which leave
    some problems unanswered (score = 0.5). See:

        A. Peñas and A. Rodrigo. A Simple Measure to Assess Nonresponse.
        In Proc. of the 49th Annual Meeting of the Association for
        Computational Linguistics, Vol. 1, pages 1415-1424, 2011.

    Parameters
    ----------
    prediction_scores : array [n_problems]

        The predictions outputted by a verification system.
        Assumes `0 >= prediction <=1`.

    ground_truth_scores : array [n_problems]

        The gold annotations provided for each problem.
        Will always be `0` or `1`.

    Returns
    ----------
    c@1 = the c@1 measure (which accounts for unanswered
        problems.)


    References
    ----------
        - E. Stamatatos, et al. Overview of the Author Identification
        Task at PAN 2014. CLEF (Working Notes) 2014: 877-897.
        - A. Peñas and A. Rodrigo. A Simple Measure to Assess Nonresponse.
        In Proc. of the 49th Annual Meeting of the Association for
        Computational Linguistics, Vol. 1, pages 1415-1424, 2011.

    """

    n = float(len(pred_y))
    nc, nu = 0.0, 0.0

    for gt_score, pred_score in zip(true_y, pred_y):
        if pred_score == 0.5:
            nu += 1
        elif (pred_score > 0.5) == (gt_score > 0.5):
            nc += 1.0
    
    return (1 / n) * (nc + (nu * nc / n))


def f1(true_y, pred_y):
    """
    Assesses verification performance, assuming that every
    `score > 0.5` represents a same-author pair decision.
    Note that all non-decisions (scores == 0.5) are ignored
    by this metric.

    Parameters
    ----------
    prediction_scores : array [n_problems]

        The predictions outputted by a verification system.
        Assumes `0 >= prediction <=1`.

    ground_truth_scores : array [n_problems]

        The gold annotations provided for each problem.
        Will typically be `0` or `1`.

    Returns
    ----------
    acc = The number of correct attributions.

    References
    ----------
        E. Stamatatos, et al. Overview of the Author Identification
        Task at PAN 2014. CLEF (Working Notes) 2014: 877-897.
    """
    true_y_filtered, pred_y_filtered = [], []

    for true, pred in zip(true_y, pred_y):
        if pred != 0.5:
            true_y_filtered.append(true)
            pred_y_filtered.append(pred)
    
    pred_y_filtered = binarize(pred_y_filtered)

    return f1_score(true_y_filtered, pred_y_filtered)


def f_05_u_score(true_y, pred_y, pos_label=1, threshold=0.5):
    """
    Return F0.5u score of prediction.

    :param true_y: true labels
    :param pred_y: predicted labels
    :param threshold: indication for non-decisions (default = 0.5)
    :param pos_label: positive class label (default = 1)
    :return: F0.5u score
    """

    pred_y = binarize(pred_y)

    n_tp = 0
    n_fn = 0
    n_fp = 0
    n_u = 0

    for i, pred in enumerate(pred_y):
        if pred == threshold:
            n_u += 1
        elif pred == pos_label and pred == true_y[i]:
            n_tp += 1
        elif pred == pos_label and pred != true_y[i]:
            n_fp += 1
        elif true_y[i] == pos_label and pred != true_y[i]:
            n_fn += 1

    return (1.25 * n_tp) / (1.25 * n_tp + 0.25 * (n_fn + n_u) + n_fp)


def load_file(fn):
    problems = {}
    for line in open(fn):
        d =  json.loads(line.strip())
        if 'value' in d:
            problems[d['id']] = d['value']
        else:
            problems[d['id']] = int(d['same'])
    return problems


def evaluate_all(true_y, pred_y):
    """
    Convenience function: calculates all PAN20 evaluation measures
    and returns them as a dict, including the 'overall' score, which
    is the mean of the individual metrics (0 >= metric >= 1). All 
    scores get rounded to three digits.
    """

    results = {'auc': auc(true_y, pred_y),
               'c@1': c_at_1(true_y, pred_y),
               'f_05_u': f_05_u_score(true_y, pred_y),
               'F1': f1(true_y, pred_y)}
    
    results['overall'] = np.mean(list(results.values()))

    for k, v in results.items():
        results[k] = round(v, 3)

    return results



# Constants
DATA_DIR = 'data/small/'
TEMP_DATA_DIR = 'temp_data/pan20_computed/'

# Load gound truth
ground_truth = {}
partition = {}

#tot = 52601/3
limited = False
tot = 20000
n_of_pos_we_want = tot/2
n_of_neg_we_want = tot/2

# quindi in totale 1_000
positive_samples = 0
negative_samples = 0
total = 0

# Qui apriamo il dataset e estraiamo un totale di 10k esempi del task
with open(DATA_DIR + '/pan20-authorship-verification-training-small-truth.jsonl', 'r') as f:
    
    for counter, l in enumerate(f):
        total += 1
        
        d = json.loads(l)

        if d['same'] and positive_samples < n_of_pos_we_want : # se la label è true e dobbiamo ancora aggiungerne
            ground_truth[d['id']] = d['same']
            positive_samples += 1
        elif not d['same'] and negative_samples < n_of_neg_we_want:
            ground_truth[d['id']] = d['same']
            negative_samples += 1
            
        # Una volta che abbiamo raggiunto il numero di dati che vogliamo, è fatta, usciamo
        if limited and (positive_samples == n_of_pos_we_want and negative_samples == n_of_neg_we_want):
            break
            
print(total, positive_samples, negative_samples)

samples = multiprocessing.Manager().Queue()

from sklearn.model_selection import train_test_split                 
def process_pair(l):
    d = json.loads(l)
    if d['id'] in ground_truth:
        e1 = prepare_entry(d['pair'][0])
        e2 = prepare_entry(d['pair'][1])
        samples.put({'id': d['id'], 'doc1': e1, 'doc2': e2})
        return {'id': d['id'], 'doc1':e1, 'doc2':e2}    


start_time = datetime.datetime.now()
print("Started handling of each pair")
with open(DATA_DIR + 'pan20-authorship-verification-training-small.jsonl', 'r') as f:
    with multiprocessing.Pool() as pool:
        samples = pool.map(process_pair, (l for l in f))
print("Ended handling of each pair")        

end_time = datetime.datetime.now()
print("Time spent: ", (end_time-start_time).total_seconds())

labels = []

# Il processo in parallelo restituisce comunque eventualmente dei None se 
# il sample non appartiene alla lista di quelli che abbiamo selezionato,
# quindi nel caso rimuoviamo i None così
samples = [i for i in samples if i is not None]

for s in samples:
    # Qui non dovrebbe arrivare
    if s is None:
        print("None")
        break
        
    # Questo dovrebbe essere il caso normale
    else:
        s['label'] = int(ground_truth[s['id']])

import pandas as pd
df = pd.DataFrame(samples)
df = df.set_index('id')
print("Start split")
X_design, X_test, y_design, y_test = train_test_split(df, df['label'], stratify=df['label'], random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_design, y_design, stratify=y_design, random_state=42)
print("End split")

print("Start picle")
# Per salvare
X_test.to_pickle(TEMP_DATA_DIR+"x_test.pkl")
X_valid.to_pickle(TEMP_DATA_DIR+"x_valid.pkl")

X_train.to_pickle(TEMP_DATA_DIR+"x_train.pkl")
df.to_pickle(TEMP_DATA_DIR+"df.pkl")
print("End picle")

start = datetime.datetime.now()
print("Start transformer")
docs = list(X_train['doc1']) + list(X_train['doc2'])
transformer = get_writeprints_transformer()
X = transformer.fit_transform(docs[:len(docs)//8]) # Usiamo 1/2 dei docs per fittare, valori più alti fanno esplodere
scaler = StandardScaler(with_mean=False)
X = scaler.fit_transform(X)

with open(TEMP_DATA_DIR + 'transformers.p', 'wb') as f:
    pickle.dump((transformer, scaler), f)

end = datetime.datetime.now()
print("End transformer")
print("Time: ", (end-start).total_seconds())

start = datetime.datetime.now()

x1 = scaler.transform(transformer.transform(X_train['doc1']))
x2 = scaler.transform(transformer.transform(X_train['doc2']))

X_train_features = pd.DataFrame(np.abs(x1-x2).todense())

end = datetime.datetime.now()
print("Time: ", (end-start).total_seconds())

start = datetime.datetime.now()

x1 = scaler.transform(transformer.transform(X_test['doc1']))
x2 = scaler.transform(transformer.transform(X_test['doc2']))

X_test_features = pd.DataFrame(np.abs(x1-x2).todense())

end = datetime.datetime.now()
print("Time: ", (end-start).total_seconds())

start = datetime.datetime.now()
        
x1 = scaler.transform(transformer.transform(X_valid['doc1']))
x2 = scaler.transform(transformer.transform(X_valid['doc2']))
X_valid_features = pd.DataFrame(np.abs(x1-x2).todense())

end = datetime.datetime.now()
print("Time: ", (end-start).total_seconds())

import pickle, pandas as pd

# Per leggere quelli già salvati
X_test_features = pd.read_pickle(TEMP_DATA_DIR+"/x_test_features.pkl")
X_valid_features = pd.read_pickle(TEMP_DATA_DIR+"/x_valid_features.pkl")
X_train_features = pd.read_pickle(TEMP_DATA_DIR+"/x_train_features.pkl")

X_test = pd.read_pickle(TEMP_DATA_DIR+"/x_test.pkl")
X_valid = pd.read_pickle(TEMP_DATA_DIR+"/x_valid.pkl")
X_train = pd.read_pickle(TEMP_DATA_DIR+"/x_train.pkl")

df = pd.read_pickle(TEMP_DATA_DIR+"/df.pkl")
y_train = X_train[['label']]
y_valid = X_valid[['label']]
y_test = X_test[['label']]

with open(TEMP_DATA_DIR + 'transformers.p', 'rb') as f:
    transformer, scaler = pickle.load(f)
    
with open(TEMP_DATA_DIR + 'ordering_metadata.p', 'rb') as f:
    train_sz, test_sz, val_sz, train_idxs = pickle.load(f)
    
with open(TEMP_DATA_DIR + 'ordering_metadata.p', 'rb') as f:
    train_idxs = pickle.load(f)

clf = LogisticRegression(random_state=42, verbose=1).fit(X_train_features, y_train)
y_pred = clf.predict(X_test_features)
y_test_new = []
for i, row in y_test.iterrows():
    y_test_new.append(row.values[0])
y_test_new = np.array(y_test_new)
#y_test = y_test_new
print(evaluate_all(y_test_new, y_pred))