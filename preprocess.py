import os
import pickle
import math
import json 
import random

from random import sample
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from tokenizer import spacy_tokenizer, sentences
# from itertools import islice

from settings.logging import printLog
from settings.static_values import EXPECTED_PREPROCESSED_DATASETS_FOLDER

def save_data_to_pickle(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

def load_data_from_pickle(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def data_exists(*filenames):
    return all(os.path.exists(filename) for filename in filenames)

def load_or_process_data(cutoff=0,sentence_size=0,no_load_flag=False,k=4,d=1,arg=None,author_id=0, dataset='none'):
    # Define pickle filenames
    printLog.debug(f'Selected AUTHOR {author_id}')
    feature_type = arg.get('feature_type')
    samples_count = arg.get('samples')
    #feature_extractor_ngram_range=arg.get('feature_extractor_ngram_range') NOT RELEVANT. IS COMPUTED AT FEATURE EXTRACTION STEP.
    special_chars = arg.get('special_chars')
    word_length_dist = arg.get('word_length_dist')
    include_vocab_richness = arg.get('include_vocab_richness')

    ra = arg.get('ra')
    ra_pcc_rate = arg.get('ra_number_of_ra_inserts')
    ra_part_size = arg.get('ra_PCC_part_size')
    incert_cc = arg.get('insert_cc')

    feature_param = str(feature_type) + str(special_chars) + str(word_length_dist) + str(include_vocab_richness) + str(samples_count) # + str(feature_extractor_ngram_range[0]) + str(feature_extractor_ngram_range[1])   
    ra_param = str(ra)+str(k)+str(d)+str(sentence_size)+str(ra_pcc_rate)+str(ra_part_size)+str(incert_cc)
    dataset_param = str(dataset)+str(author_id)
    
    root_path = EXPECTED_PREPROCESSED_DATASETS_FOLDER
    os.makedirs(EXPECTED_PREPROCESSED_DATASETS_FOLDER, exist_ok=True)
    x_train_pickle = root_path+str(cutoff)+str(dataset_param)+str(feature_param)+str(ra_param)+'-x_train.pkl'
    y_train_pickle = root_path+str(cutoff)+str(dataset_param)+str(feature_param)+str(ra_param)+'-y_train.pkl'
    x_test_pickle = root_path+str(cutoff)+str(dataset_param)+str(feature_param)+str(ra_param)+'-x_test.pkl'
    y_test_pickle = root_path+str(cutoff)+str(dataset_param)+str(feature_param)+str(ra_param)+'-y_test.pkl'
    if ra:
        pcc_test_pickle = root_path+str(cutoff)+str(dataset_param)+str(feature_param)+str(ra_param)+'-pcc_test.pkl'
        pcc_train_pickle = root_path+str(cutoff)+str(dataset_param)+str(feature_param)+str(ra_param)+'-pcc_train.pkl'
        raw_c_test_pickle = root_path+str(cutoff)+str(dataset_param)+str(feature_param)+str(ra_param)+'-raw_c_test.pkl'
        raw_c_train_pickle = root_path+str(cutoff)+str(dataset_param)+str(feature_param)+str(ra_param)+'-raw_c_train.pkl'
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
            pcc_test_params = load_data_from_pickle(pcc_test_pickle)
            pcc_train_params = load_data_from_pickle(pcc_train_pickle) 
            raw_c_train = load_data_from_pickle(raw_c_train_pickle)
            raw_c_test = load_data_from_pickle(raw_c_test_pickle)
        else:
            pcc_train_params = []
            pcc_test_params = []
            raw_c_train = []
            raw_c_test = []
        printLog.debug('Preprocessed data loaded from pickle files')
    else:
        # Your data processing/loading function here
        x_train, y_train, x_test, y_test, raw_c_train, raw_c_test, pcc_train_params, pcc_test_params = load_corpus(_cutoff=cutoff,sentence_size=sentence_size,k=k,d=d,arg=arg,author_id=author_id)
        #x_train, y_train, x_test, y_test, pcc_train_params, pcc_test_params = load_corpus(_cutoff=cutoff,sentence_size=sentence_size,k=k,d=d,arg=arg)
        if ra:
            printLog.debug(f'sizes: x_train: {len(x_train)}, y_train: {len(y_train)}, x_test: {len(x_test)}, y_test: {len(y_test)}, raw_c_train: {len(raw_c_train)}, raw_c_test: {len(raw_c_test)}')
        
        # After processing, save the datasets to pickle files
        save_data_to_pickle(x_train, x_train_pickle)
        save_data_to_pickle(y_train, y_train_pickle)
        save_data_to_pickle(x_test, x_test_pickle)
        save_data_to_pickle(y_test, y_test_pickle)
        if ra:
            save_data_to_pickle(pcc_train_params, pcc_train_pickle)
            save_data_to_pickle(pcc_test_params, pcc_test_pickle)
            save_data_to_pickle(raw_c_train, raw_c_train_pickle)
            save_data_to_pickle(raw_c_test, raw_c_test_pickle)
        else:
            pcc_train_params = []
            pcc_test_params = []
            raw_c_train = []
            raw_c_test = []
        printLog.debug('Preprocessed data saved to pickle files')
    
    return x_train, y_train, x_test, y_test, raw_c_train, raw_c_test, pcc_train_params, pcc_test_params


def process_pair(args):
    i, pair, y_train, cc_array, pcc_samples, sentence_size,k,d,arg = args
    
    kt = pair[0]
    ut = pair[1]
    s_ut = sentences(ut, sentence_size)
    s_kt = sentences(kt, sentence_size)
    
    x_out = []
    y_out = []

    insert_cc = arg.get('insert_cc')
    if cc_array and insert_cc:
        Y = int(arg.get('ra_PCC_part_size'))
        s_pcc_sample = sentences(pcc_samples, sentence_size)
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
    
    pcc_params = {
        'N': N,
        'k': k,
        'd': d,
        'n': n,
        'l': l,
        'w': w,
        'l_group': l_group
    }
    return x_out, y_out, c, pcc_params

def rolling_selection(x_train, y_train, cc_array, pcc_samples, sentence_size, k=4, d=1, arg=None):
    """
    k - window size\n
    d - overlap window
    """
    printLog.debug(f'Rolling selection - sentence size: {sentence_size}, k: {k}, d: {d}')

    # for i, pair in enumerate(x_train):
    #     process_pair((i, pair, y_train[i], cc_array[i], pcc_samples[i], sentence_size,k,d))

    with Pool() as pool:
        results = pool.map(process_pair, [(i, pair, y_train[i], cc_array[i], pcc_samples[i], sentence_size,k,d, arg) for i, pair in enumerate(x_train)])

    x_out = []
    y_out = []
    raw_c = []
    pcc_params = []

    for x, y, raw_c_value, PCC_p in results:
        x_out.extend(x)
        raw_c.extend(raw_c_value)
        y_out.extend(y)
        pcc_params.append(PCC_p)

    return x_out, y_out, raw_c, pcc_params

def load_corp(x_path, y_path, pcc_samples_path, sentence_size=0, cutoff=0,cc_flag=False,k=4,d=1,arg=None):
    # Read the ground truth:
    # Read and process y
    y = {}
    author = {}
    pcc_rate = arg.get('ra_number_of_ra_inserts')
    
    for line in open(y_path, encoding='utf-8'):
        y_json_output = json.loads(line.strip())
        y[y_json_output['id']] = y_json_output['same']
        author[y_json_output['id']] = y_json_output['authors']
        
    if cutoff and len(y) > cutoff:
        true_items = [(k, v) for k, v in y.items() if v is True]
        false_items = [(k, v) for k, v in y.items() if v is False]

        cutoff_half = cutoff // 2  # Integer division to get half of the cutoff

        # Ensure there are enough True and False items
        if len(true_items) < cutoff_half or len(false_items) < cutoff_half:
            printLog.error(f"Not enough True or False items to meet the cutoff. True Items: {len(true_items)}, False Items: {len(false_items)}, Cutoff half: {cutoff_half}")
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
    for line in tqdm(open(x_path, encoding='utf-8')):
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
         
    pcc_samples = []

    for line in open(pcc_samples_path, encoding='utf-8'):
        pcc_json = json.loads(line.strip())
        if any(author in str(pcc_json['author']) for author in unique_list_author):
            continue
        elif len(pcc_samples) >= len(x_filtered):
            break
        else:
            pcc_samples.append(pcc_json['text'])

    cc_array = [1 if classification == 1 and i < len(y_filtered)*(pcc_rate)/2 else 0 for i, classification in enumerate(y_filtered)]
    printLog.debug(f'Post-processing sizes - x: {len(x_filtered)}, y: {len(y_filtered)}')
    
    if cc_flag: # contract cheating flag
        x_filtered, y_filtered, raw_c, pcc_params = rolling_selection(x_filtered, y_filtered, cc_array, pcc_samples, sentence_size,k=k,d=d, arg=arg)
    else: 
        pcc_params = []
        raw_c = 0
    printLog.debug(f'Post-splitting sizes - x: {len(x_filtered)}, y: {len(y_filtered)}')

    printLog.debug('Tokenizing using spaCy')
    with Pool(cpu_count() - 1) as pool:
        x_filtered = pool.map(spacy_tokenizer, [(x, arg) for x in x_filtered])
    printLog.debug('Tokenizing complete')

    printLog.debug(f'Size of y {len(y_filtered)}')

    return x_filtered, y_filtered, raw_c, pcc_params

def load_corpus(_cutoff=0,sentence_size=0,k=4,d=1,arg=None,author_id=0):
    dataset = arg.get('dataset')
    x_train_path = dataset.get('dataset_train_path_pair')
    y_train_path = dataset.get('dataset_train_path_truth')
    x_test_path = dataset.get('dataset_test_path_pair')
    y_test_path = dataset.get('dataset_test_path_truth')

    if author_id != '0':
        x_train_path = x_train_path.replace("AUTHOR", str(author_id))
        y_train_path = y_train_path.replace("AUTHOR", str(author_id))
        x_test_path = x_test_path.replace("AUTHOR", str(author_id))
        y_test_path = y_test_path.replace("AUTHOR", str(author_id))
    
    ra = arg.get('ra')
    if ra:
        pcc_train_samples = dataset['pcc_train_samples']    # f"{root}/datasets/pan20-created/pan20-train-all-different-authors.jsonl"
        pcc_test_samples = dataset['pcc_test_samples']      # f"{root}/datasets/pan20-created/pan20-test-all-different-authors.jsonl"
    else:
        pcc_train_samples = pcc_test_samples = "../datasets/pan20-created/pan20-train-all-different-authors.jsonl"

    printLog.debug('Loading and extracting features')
    x_train, y_train, raw_c_train, pcc_train_params = load_corp(x_train_path, y_train_path, pcc_train_samples, cutoff=_cutoff,cc_flag=ra,sentence_size=sentence_size,k=k,d=d,arg=arg)
    x_test, y_test, raw_c_test, pcc_test_params = load_corp(x_test_path, y_test_path,pcc_test_samples,cutoff=_cutoff,cc_flag=ra,sentence_size=sentence_size,k=k,d=d,arg=arg)
    
    return x_train, y_train, x_test, y_test, raw_c_train, raw_c_test, pcc_train_params, pcc_test_params


