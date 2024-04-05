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
        raw_c_test_pickle = root_path+str(cutoff)+str(feature_param)+str(ra_param)+'-raw_c_test.pkl'
        raw_c_train_pickle = root_path+str(cutoff)+str(feature_param)+str(ra_param)+'-raw_c_train.pkl'
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
            raw_c_train = load_data_from_pickle(raw_c_train_pickle)
            raw_c_test = load_data_from_pickle(raw_c_test_pickle)
        else:
            PCC_train_params = []
            PCC_test_params = []
            raw_c_train = []
            raw_c_test = []
        printLog.debug('Preprocessed data loaded from pickle files')
    else:
        # Your data processing/loading function here
        x_train, y_train, x_test, y_test, raw_c_train, raw_c_test, PCC_train_params, PCC_test_params = load_corpus(_cutoff=cutoff,sentence_size=sentence_size,k=k,d=d,arg=arg)
        #x_train, y_train, x_test, y_test, PCC_train_params, PCC_test_params = load_corpus(_cutoff=cutoff,sentence_size=sentence_size,k=k,d=d,arg=arg)
        printLog.debug(f'sizes: x_train: {len(x_train)}, y_train: {len(y_train)}, x_test: {len(x_test)}, y_test: {len(y_test)}, raw_c_train: {len(raw_c_train)}, raw_c_test: {len(raw_c_test)}')
        
        # After processing, save the datasets to pickle files
        save_data_to_pickle(x_train, x_train_pickle)
        save_data_to_pickle(y_train, y_train_pickle)
        save_data_to_pickle(x_test, x_test_pickle)
        save_data_to_pickle(y_test, y_test_pickle)
        if ra:
            save_data_to_pickle(PCC_train_params, pcc_train_pickle)
            save_data_to_pickle(PCC_test_params, pcc_test_pickle)
            save_data_to_pickle(raw_c_train, raw_c_train_pickle)
            save_data_to_pickle(raw_c_test, raw_c_test_pickle)
        else:
            PCC_train_params = []
            PCC_test_params = []
            raw_c_train = []
            raw_c_test = []
        printLog.debug('Preprocessed data saved to pickle files')
    
    return x_train, y_train, x_test, y_test, raw_c_train, raw_c_test, PCC_train_params, PCC_test_params


def process_pair(args):
    i, pair, y_train, cc_array, PCC_samples, sentence_size,k,d,arg = args
    
    kt = pair[0]
    ut = pair[1]
    s_ut = sentences(ut, sentence_size)
    s_kt = sentences(kt, sentence_size)
    
    x_out = []
    y_out = []

    if cc_array:
        Y = int(arg.get('ra_PCC_part_size'))
        s_pcc_sample = sentences(PCC_samples, sentence_size)
        c = [y_train]*len(s_ut)
        
        random_index_c = random.randint(0, len(c) - 1)

        # Shuffle s_pcc_sample and take the first N elements
        random.shuffle(s_pcc_sample)
        selected_samples = s_pcc_sample[:Y]

        for i, sample in enumerate(selected_samples):
            s_ut.insert(random_index_c + i, sample)
            c.insert(random_index_c + i, 0)
    else:
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
    return x_out, y_out, c, PCC_params

def rolling_selection(x_train, y_train, cc_array, PCC_samples, sentence_size, k=4, d=1, arg=None):
    """
    k - window size\n
    d - overlap window
    """
    printLog.debug(f'Selecting text - sentence size: {sentence_size}, k: {k}, d: {d}')

    # for i, pair in enumerate(x_train):
    #     process_pair((i, pair, y_train[i], cc_array[i], PCC_samples[i], sentence_size,k,d))

    with Pool() as pool:
        results = pool.map(process_pair, [(i, pair, y_train[i], cc_array[i], PCC_samples[i], sentence_size,k,d, arg) for i, pair in enumerate(x_train)])

    x_out = []
    y_out = []
    raw_c = []
    PCC_params = []

    for x, y, raw_c_value, PCC_p in results:
        x_out.extend(x)
        raw_c.extend(raw_c_value)
        y_out.extend(y)
        PCC_params.append(PCC_p)

    return x_out, y_out, raw_c, PCC_params

def load_corp(x_path, y_path, PCC_samples_path, sentence_size=0, cutoff=0,cc_flag=False,k=4,d=1,arg=None):
    # Read the ground truth:
    # Read and process y
    y = {}
    author = {}
    pcc_rate = 0.3
    
    for line in open(y_path):
        y_json_output = json.loads(line.strip())
        y[y_json_output['id']] = y_json_output['same']
        author[y_json_output['id']] = y_json_output['authors']
        
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

    

    # Initialize x dictionary
    x_dict = {}
    author_dict = {}
    for line in tqdm(open(x_path)):
        json_output = json.loads(line.strip())
        if json_output['id'] in y:
            x_dict[json_output['id']] = json_output['pair']
            author_dict[json_output['id']] = author[json_output['id']]
    
    # Filter x based on y keys to ensure matching
    x_filtered = [x_dict[id] for id in y if id in x_dict]
    author_filtered = [author_dict[id] for id in y if id in x_dict]
    y_filtered = [int(y[id]) for id in y if id in x_dict]


    flattened_list_author = [item for sublist in author_filtered for item in sublist]
    unique_list_author = list(set(flattened_list_author))
    
        
    PCC_samples = []

    for line in open(PCC_samples_path):
        PCC_json = json.loads(line.strip())
        if any(author in str(PCC_json['author']) for author in unique_list_author):
            continue
        elif len(PCC_samples) >= len(x_filtered):
            break
        else:
            PCC_samples.append(PCC_json['text'])

    cc_array = [1 if classification == 1 and i < len(y_filtered)*(pcc_rate)/2 else 0 for i, classification in enumerate(y_filtered)]

    printLog.debug(f'Post-processing sizes - x: {len(x_filtered)}, y: {len(y_filtered)}')
    
    if cc_flag: # contract cheating flag
        x_filtered, y_filtered, raw_c, PCC_params = rolling_selection(x_filtered, y_filtered, cc_array, PCC_samples, sentence_size,k=k,d=d, arg=arg)
    else: 
        PCC_params = []
        raw_c = 0
    printLog.debug(f'Post-splitting sizes - x: {len(x_filtered)}, y: {len(y_filtered)}')

    printLog.debug('Tokenizing using spaCy')
    with Pool(cpu_count() - 1) as pool:
        x_filtered = pool.map(spacy_tokenizer, [(x, arg) for x in x_filtered])
    printLog.debug('Tokenizing complete')

    printLog.debug(f'Size of y {len(y_filtered)}')

    return x_filtered, y_filtered, raw_c, PCC_params

def load_corpus(_cutoff=0,sentence_size=0,k=4,d=1,arg=None):
    ra = arg.get('ra')
    # x_train_path = "../datasets/pan20-authorship-verification-training-small/pan20-authorship-verification-training-small.jsonl"
    # y_train_path = "../datasets/pan20-authorship-verification-training-small/pan20-authorship-verification-training-small-truth.jsonl"
    # x_test_path = "../datasets/pan20-authorship-verification-test/pan20-authorship-verification-test.jsonl"
    # y_test_path = "../datasets/pan20-authorship-verification-test/pan20-authorship-verification-test-truth.jsonl"
    x_train_path = "../datasets/pan20-created/pan20-train-pairs-1000555.jsonl"
    y_train_path = "../datasets/pan20-created/pan20-train-truth-1000555.jsonl"
    PCC_train_samples = "../datasets/pan20-created/pan20-train-all-different-authors.jsonl"
    x_test_path = "../datasets/pan20-created/pan20-test-pairs-1000555.jsonl"
    y_test_path = "../datasets/pan20-created/pan20-test-truth-1000555.jsonl"
    PCC_test_samples = "../datasets/pan20-created/pan20-test-all-different-authors.jsonl"

    printLog.debug('Loading and extracting features')
    x_train, y_train, raw_c_train, PCC_train_params = load_corp(x_train_path, y_train_path, PCC_train_samples, cutoff=_cutoff,cc_flag=ra,sentence_size=sentence_size,k=k,d=d,arg=arg)
    x_test, y_test, raw_c_test, PCC_test_params = load_corp(x_test_path, y_test_path,PCC_test_samples,cutoff=_cutoff,cc_flag=ra,sentence_size=sentence_size,k=k,d=d,arg=arg)
    
    return x_train, y_train, x_test, y_test, raw_c_train, raw_c_test, PCC_train_params, PCC_test_params


