#!/usr/bin/python3
# encoding: utf-8

from settings.arguments import parse_arguments_controller
from steps.setup import setup
from steps.load_json_file import read_jsonl

from settings.static_values import EXPECTED_PREPROCESSED_DATASETS_FOLDER, EXPECTED_PREPROCESSED_DATASET_FILES


import argparse
import json
import random
import os
import glob
import shutil
from itertools import combinations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
from seaborn import kdeplot
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from nltk import tokenize, pos_tag

tokenizer = tokenize.TreebankWordTokenizer()

def feature_worker(pair, feature=None):
    kt = feature_extractor(pair[0])
    ut = feature_extractor(pair[1])
    return kt, ut

def feature_extractor(text, feature=None):
    tokens = []
    for token in tokenizer.tokenize(text):
        tokens.append(token)
    
    tagger_output = pos_tag(tokens)
    pos_tags = [token[1] for token in tagger_output]
   
    entry = {
        'preprocessed': text,
        'pos_tags': pos_tags,
        'tokens': tokens
    }
    return entry

def custom_vectorizer(vectorizer, ngram_size, vocab_size):
    if vectorizer == 'tfidf-char':
        return TfidfVectorizer(analyzer='char', ngram_range=(ngram_size, ngram_size), max_features=vocab_size)
    else:
        raise ValueError('Unknown vectorizer: {}'.format(vectorizer))

def main():
    parser = argparse.ArgumentParser(description='Authorship Verification - Master Thesis')

    # data settings:
    parser.add_argument('-input_pairs', type=str, required=True,
                        help='Path to the jsonl-file with the input pairs')
    parser.add_argument('-input_truth', type=str, required=True,
                        help='Path to the ground truth-file for the input pairs')
    parser.add_argument('-test_pairs', type=str, required=True,
                        help='Path to the jsonl-file with the test pairs')
    parser.add_argument('-test_truth', type=str, required=True, 
                        help='Path to the ground truth-file for the test pairs')
    parser.add_argument('-output', type=str, required=True,
                        help='Path to the output folder for the predictions.\
                             (Will be overwritten if it exist already.)')

    # algorithmic settings:
    parser.add_argument('-seed', default=2020, type=int,
                        help='Random seed')
    parser.add_argument('-vocab_size', default=3000, type=int,
                        help='Maximum number of vocabulary items in feature space')
    parser.add_argument('-ngram_size', default=4, type=int,
                        help='Size of the ngrams')
    parser.add_argument('-num_iterations', default=0, type=int,
                        help='Number of iterations (`k`); zero by default')
    parser.add_argument('-dropout', default=.5, type=float,
                        help='Proportion of features to keep in each iteration')

    args = parser.parse_args()
    print(args)

    np.random.seed(args.seed)
    random.seed(args.seed)

    try:
        shutil.rmtree(args.output)
    except FileNotFoundError:
        pass
    os.mkdir(args.output)

    # Read the ground truth:
    truth = {}
    for line in open(args.input_truth):
        json_output = json.loads(line.strip())
        truth[json_output['id']] = json_output['same']
    
    # Truncation of the input pairs:
    cutoff = 0 
    if cutoff:
        truth = dict(random.sample(truth.items(), cutoff))
        print(len(truth))
    
    # Read the input pairs:
    print('-> feature extraction')

    input_pairs = {}
    for line in tqdm(open(args.input_pairs)):
        json_output = json.loads(line.strip())
        if json_output['id'] in truth:
            with Pool(cpu_count() - 1) as pool: 
                kt, ut = pool.map(feature_worker,json_output['pair'])
            input_pairs.extend([kt, ut])
    
    print()
    
    

    print('-> constructing vectorizer')
    vectorizer = custom_vectorizer('tfidf-char', args.ngram_size, args.vocab_size)
    
    print('-> fitting vectorizer to input pairs')
    vectorizer.fit(input_pairs)

    print('-> Grid search to find optimal threshold')

    print('-> Transform vectorizer')
    x1,x2 = vectorizer.transform(input_pairs).toarray()

    print('-> Compute similarities')

    print('-> Classify input pairs')

    print('-> Read test pairs and do predictions')

    print('-> Evaluate predictions')

if __name__ == '__main__':
    main()