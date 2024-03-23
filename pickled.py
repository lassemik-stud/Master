import os
import pickle

from itertools import islice
import random
from random import sample
import json 
import spacy
nlp = spacy.load('en_core_web_sm')
from tqdm import tqdm
from settings.logging import printLog
from multiprocessing import Pool, cpu_count

def save_data_to_pickle(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

def load_data_from_pickle(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def data_exists(*filenames):
    return all(os.path.exists(filename) for filename in filenames)

def load_or_process_data(cutoff=0,sentence_size=0,no_load_flag=False):
    # Define pickle filenames
    root_path = '../pre_data/'
    x_train_pickle = root_path+str(cutoff)+'-x_train.pkl'
    y_train_pickle = root_path+str(cutoff)+'-y_train.pkl'
    x_test_pickle = root_path+str(cutoff)+'-x_test.pkl'
    y_test_pickle = root_path+str(cutoff)+'-y_test.pkl'
    
    # Check if pickle files exist
    if data_exists(x_train_pickle, y_train_pickle, x_test_pickle, y_test_pickle) or not no_load_flag:
        # Load data from pickle files
        x_train = load_data_from_pickle(x_train_pickle)
        y_train = load_data_from_pickle(y_train_pickle)
        x_test = load_data_from_pickle(x_test_pickle)
        y_test = load_data_from_pickle(y_test_pickle)
        printLog.debug('Data loaded from pickle files')
    else:
        # Your data processing/loading function here
        x_train, y_train, x_test, y_test = load_corpus(cutoff,sentence_size)
        printLog.debug(f'sizes: x_train: {len(x_train)}, y_train: {len(y_train)}, x_test: {len(x_test)}, y_test: {len(y_test)}')
        
        # After processing, save the datasets to pickle files
        save_data_to_pickle(x_train, x_train_pickle)
        save_data_to_pickle(y_train, y_train_pickle)
        save_data_to_pickle(x_test, x_test_pickle)
        save_data_to_pickle(y_test, y_test_pickle)
        printLog.debug('Data saved to pickle files')
    
    return x_train, y_train, x_test, y_test

def sentences(text, n):
    sents = [i.text for i in nlp(text).sents]
    return [' '.join(sents[i:i+n]) for i in range(0, len(sents), n)]

def sentences(text, n):
    sents = [i.text for i in nlp(text).sents]
    return [' '.join(sents[i:i+n]) for i in range(0, len(sents), n)]

def process_pair(args):
    i, pair, y_train, sentence_size = args
    kt = pair[0]
    ut = pair[1]
    s_ut = sentences(ut, sentence_size)
    s_kt = sentences(kt, sentence_size)

    x_out = []
    y_out = []

    for sentence_kt in s_kt:
        for sentence_ut in s_ut:
            x_out.append([sentence_kt, sentence_ut])
            y_out.append(y_train[i])
    
    return x_out, y_out

def split_text(x_train, y_train, sentence_size):
    printLog.debug(f'Splitting text - sentence size is {sentence_size}')

    with Pool() as pool:
        results = pool.map(process_pair, [(i, pair, y_train, sentence_size) for i, pair in enumerate(x_train)])

    x_out = []
    y_out = []

    for x, y in results:
        x_out.extend(x)
        y_out.extend(y)

    return x_out, y_out

def load_corp(x_path, y_path, sentence_size=0, cutoff=0,cc_flag=False):
    # Read the ground truth:
    # Read and process y
    y = {}
    for line in open(y_path):
        json_output = json.loads(line.strip())
        y[json_output['id']] = json_output['same']
    
    if cutoff and len(y) > cutoff:
        true_items = [(k, v) for k, v in y.items() if v is True]
        false_items = [(k, v) for k, v in y.items() if v is False]

        cutoff_half = cutoff // 2  # Integer division to get half of the cutoff

        # Ensure there are enough True and False items
        if len(true_items) < cutoff_half or len(false_items) < cutoff_half:
            print("Not enough True or False items to meet the cutoff.")
            exit()

        sampled_true_items = sample(true_items, cutoff_half)
        sampled_false_items = sample(false_items, cutoff_half)
        
        random.seed(42)
        items = sampled_true_items + sampled_false_items
        random.shuffle(items)
        y = dict(items)

    printLog.debug(f'Size of y {len(y)}')

    # Initialize x dictionary
    x_dict = {}
    for line in tqdm(open(x_path)):
        json_output = json.loads(line.strip())
        if json_output['id'] in y:
            x_dict[json_output['id']] = json_output['pair']
    
    # Filter x based on y keys to ensure matching
    x_filtered = [x_dict[id] for id in y if id in x_dict]
    y_filtered = [int(y[id]) for id in y if id in x_dict]

    printLog.debug(f'Post-processing sizes - x: {len(x_filtered)}, y: {len(y_filtered)}')

    if cc_flag: # contract cheating flag
        x_filtered, y_filtered = split_text(x_filtered, y_filtered, sentence_size)
    printLog.debug(f'Post-splitting sizes - x: {len(x_filtered)}, y: {len(y_filtered)}')

    printLog.debug('Tokenizing using spaCy')
    with Pool(cpu_count() - 1) as pool:
        x_filtered = pool.map(spacy_tokenizer, x_filtered)
    printLog.debug('Tokenizing complete')

    return x_filtered, y_filtered

def load_corpus(_cutoff=0,sentence_size=0):
    x_train_path = "../datasets/pan20-authorship-verification-training-small/pan20-authorship-verification-training-small.jsonl"
    y_train_path = "../datasets/pan20-authorship-verification-training-small/pan20-authorship-verification-training-small-truth.jsonl"
    x_test_path = "../datasets/pan20-authorship-verification-test/pan20-authorship-verification-test.jsonl"
    y_test_path = "../datasets/pan20-authorship-verification-test/pan20-authorship-verification-test-truth.jsonl"

    printLog.debug('Loading and extracting features')
    x_train, y_train = load_corp(x_train_path, y_train_path, cutoff=_cutoff,cc_flag=True,sentence_size=sentence_size)
    x_test, y_test = load_corp(x_test_path, y_test_path,cutoff=_cutoff,cc_flag=True,sentence_size=sentence_size)
    
    return x_train, y_train, x_test, y_test

# Tokenizer function that uses spaCy for lemmatization
def spacy_tokenizer(pair):
    kt = pair[0]
    ut = pair[1]
    kt_res = " ".join([token.lemma_ for token in nlp(kt)])
    ut_res = " ".join([token.lemma_ for token in nlp(ut)])
    return [kt_res, ut_res]
