import os
import pickle

import math
from itertools import islice
import random
from random import sample
import json 


from tqdm import tqdm
from settings.logging import printLog
from multiprocessing import Pool, cpu_count

from tokenizer import spacy_tokenizer, sentences

def save_data_to_pickle(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

def load_data_from_pickle(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def data_exists(*filenames):
    return all(os.path.exists(filename) for filename in filenames)

def load_or_process_data(cutoff=0,sentence_size=0,no_load_flag=False,k=4,d=1,arg=None):
    # Define pickle filenames
    feature_type = arg.get('feature_type')
    special_chars = arg.get('special_chars')
    word_length_dist = arg.get('word_length_dist')
    include_vocab_richness = arg.get('include_vocab_richness')

    ra = arg.get('ra')

    feature_param = str(feature_type) + str(special_chars) + str(word_length_dist) + str(include_vocab_richness)
    ra_param = str(ra)+str(k)+str(d)+str(sentence_size)
    
    root_path = '../pre_data/'
    x_train_pickle = root_path+str(cutoff)+str(feature_param)+str(ra_param)+'-x_train.pkl'
    y_train_pickle = root_path+str(cutoff)+str(feature_param)+str(ra_param)+'-y_train.pkl'
    x_test_pickle = root_path+str(cutoff)+str(feature_param)+str(ra_param)+'-x_test.pkl'
    y_test_pickle = root_path+str(cutoff)+str(feature_param)+str(ra_param)+'-y_test.pkl'
    if ra:
        pcc_test_pickle = root_path+str(cutoff)+str(feature_param)+str(ra_param)+'-pcc_test.pkl' 
        pcc_train_pickle = root_path+str(cutoff)+str(feature_param)+str(ra_param)+'-pcc_train.pkl' 
    #print(pcc_test_pickle)
    #print(pcc_train_pickle)
    
    # Check if pickle files exist
    if data_exists(x_train_pickle, y_train_pickle, x_test_pickle, y_test_pickle):
        # Load data from pickle files
        x_train = load_data_from_pickle(x_train_pickle)
        y_train = load_data_from_pickle(y_train_pickle)
        x_test = load_data_from_pickle(x_test_pickle)
        y_test = load_data_from_pickle(y_test_pickle)
        if ra:
            PCC_test_params = load_data_from_pickle(pcc_test_pickle)
            PCC_train_params = load_data_from_pickle(pcc_train_pickle) 
        else:
            PCC_train_params = []
            PCC_test_params = []
        printLog.debug('Preprocessed data loaded from pickle files')
    else:
        # Your data processing/loading function here
        x_train, y_train, x_test, y_test, PCC_train_params, PCC_test_params = load_corpus(_cutoff=cutoff,sentence_size=sentence_size,k=k,d=d,arg=arg)
        #x_train, y_train, x_test, y_test, PCC_train_params, PCC_test_params = load_corpus(_cutoff=cutoff,sentence_size=sentence_size,k=k,d=d,arg=arg)
        printLog.debug(f'sizes: x_train: {len(x_train)}, y_train: {len(y_train)}, x_test: {len(x_test)}, y_test: {len(y_test)}')
        
        # After processing, save the datasets to pickle files
        save_data_to_pickle(x_train, x_train_pickle)
        save_data_to_pickle(y_train, y_train_pickle)
        save_data_to_pickle(x_test, x_test_pickle)
        save_data_to_pickle(y_test, y_test_pickle)
        if ra:
            save_data_to_pickle(PCC_train_params, pcc_train_pickle)
            save_data_to_pickle(PCC_test_params, pcc_test_pickle)
        else:
            PCC_train_params = []
            PCC_test_params = []
        printLog.debug('Preprocessed data saved to pickle files')
    
    return x_train, y_train, x_test, y_test, PCC_train_params, PCC_test_params


def process_pair(args):
    i, pair, y_train, sentence_size,k,d = args
 
    kt = pair[0]
    ut = pair[1]
    s_ut = sentences(ut, sentence_size)
    s_kt = sentences(kt, sentence_size)

    x_out = []
    y_out = []

    c = [y_train]*len(s_ut)

    # ROLLING ATTRIBUTION PARAMETERS
    N = len(s_ut)
    #k = 4 
    #d = 2
    n = math.ceil((N-k)/(k-d)) + 1
    l = len(s_kt)*n 
    w = k-d
    l_group = len(s_kt)

    # ROLLING SELECTION
    for j, sentence_kt in enumerate(s_kt):
        i = 0
        while i < n*w:
            flat_list = [item for sublist in s_ut[i:i+k] for item in sublist]
            x_out.append([sentence_kt, ''.join(flat_list)])
            y_out.append(0 if any(element == 0 for element in c[i:i+k]) else 1)
            i+=w
    
    PCC_params = {
        'N': N,
        'k': k,
        'd': d,
        'n': n,
        'l': l,
        'w': w,
        'l_group': l_group
    }
    return x_out, y_out, PCC_params

def rolling_selection(x_train, y_train, sentence_size, k=4, d=1):
    """
    k - window size\n
    d - overlap window
    """
    printLog.debug(f'Selecting text - sentence size is {sentence_size}')

    with Pool() as pool:
        results = pool.map(process_pair, [(i, pair, y_train[i], sentence_size,k,d) for i, pair in enumerate(x_train)])

    x_out = []
    y_out = []
    PCC_params = []

    for x, y,PCC_p in results:
        x_out.extend(x)
        y_out.extend(y)
        PCC_params.append(PCC_p)

    return x_out, y_out, PCC_params

def load_corp(x_path, y_path, sentence_size=0, cutoff=0,cc_flag=False,k=4,d=1,arg=None):
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
    y_filtered_old = y_filtered
    
    if cc_flag: # contract cheating flag
        x_filtered, y_filtered, PCC_params = rolling_selection(x_filtered, y_filtered, sentence_size,k=k,d=d)
    else: 
        PCC_params = []
    printLog.debug(f'Post-splitting sizes - x: {len(x_filtered)}, y: {len(y_filtered)}')

    printLog.debug('Tokenizing using spaCy')
    with Pool(cpu_count() - 1) as pool:
        x_filtered = pool.map(spacy_tokenizer, [(x, arg) for x in x_filtered])
    printLog.debug('Tokenizing complete')

    return x_filtered, y_filtered, PCC_params

def load_corpus(_cutoff=0,sentence_size=0,k=4,d=1,arg=None):
    ra = arg.get('ra')
    x_train_path = "../datasets/pan20-authorship-verification-training-small/pan20-authorship-verification-training-small.jsonl"
    y_train_path = "../datasets/pan20-authorship-verification-training-small/pan20-authorship-verification-training-small-truth.jsonl"
    x_test_path = "../datasets/pan20-authorship-verification-test/pan20-authorship-verification-test.jsonl"
    y_test_path = "../datasets/pan20-authorship-verification-test/pan20-authorship-verification-test-truth.jsonl"

    printLog.debug('Loading and extracting features')
    x_train, y_train, PCC_train_params = load_corp(x_train_path, y_train_path, cutoff=_cutoff,cc_flag=ra,sentence_size=sentence_size,k=k,d=d,arg=arg)
    x_test, y_test, PCC_test_params = load_corp(x_test_path, y_test_path,cutoff=_cutoff,cc_flag=ra,sentence_size=sentence_size,k=k,d=d,arg=arg)
    
    return x_train, y_train, x_test, y_test, PCC_train_params, PCC_test_params


