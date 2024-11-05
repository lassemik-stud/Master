# Only for SVM and LR. NB has different parameters.

import itertools
import numpy as np

from settings.static_values import EXPECTED_DATASETS_FOLDER
from dataset_changes.pan20_create_baseline_1 import DATASET_CREATE_PATH

name = 'default'

DATASET = {
    'dataset' : 'rolling-attribution-1',
    'dataset_train_path_pair' : EXPECTED_DATASETS_FOLDER + DATASET_CREATE_PATH + "/pan20-train-pairs-AUTHOR.jsonl",
    'dataset_train_path_truth': EXPECTED_DATASETS_FOLDER + DATASET_CREATE_PATH + "/pan20-train-truth-AUTHOR.jsonl",
    'dataset_test_path_pair' : EXPECTED_DATASETS_FOLDER + DATASET_CREATE_PATH + "/pan20-test-pairs-AUTHOR.jsonl",
    'dataset_test_path_truth' : EXPECTED_DATASETS_FOLDER + DATASET_CREATE_PATH + "/pan20-test-truth-AUTHOR.jsonl",
    'pcc_train_samples' : EXPECTED_DATASETS_FOLDER +  "/pan20-created/pan20-train-all-different-authors.jsonl",
    'pcc_test_samples' : EXPECTED_DATASETS_FOLDER + "pan20-created/pan20-test-all-different-authors.jsonl"
}
clf = {
        'SVM' : [True],
        'LR' : [False],
        'NaiveBayes' : [False]
    }

svm_parameters = {
            'svm_c': [10],
            'svm_degree': [1],
            'svm_kernel': ['poly']
        }
lr_parameters = {
            'lr_c': [0.0],
            'lr_penalty': ['Null'],
            'lr_solver': ['Null'],
            'lr_l1_ratio': [0.0],
            'lr_max_iter': [0]
        }

naiveBayes_parameters = {
            'nb_alpha': [0],
            'nb_fit_prior': [0]
        }

# AUTHOR PARAMS
AUTHOR_ID = [2049660, 3107154, 4483094]
#     \item     1.0           2049660
#     \item     0.8999        3107154
#     \item     0.8099        4483094

# AUTHOR PARAMS
AUTHOR_ID = [2049660] #, 1648312, 1777261]
#     \item     1.0       2049660
#     \item     0.909     1648312
#     \item     0.8       1777261

# BASE PARAMETERS
ra = [True]
ra_k = [3]                  # Window size
ra_d = [2]                # Overlap size
ra_sentence_size = [80]     # Size of part used to split up the text. This could be a paragraph size. Size is in number of sentences. 
samples = [1000]             # Max number of samples used in train and test
ra_PCC_part_size = [1,2,3]      # number of parts of sentence size inserted into the UT
number_of_ra_inserts = [1,2,3]
insert_cc = [True]

special_char = [True]
word_length_dist = [False]
include_vocab_richness = [False]

parameters_tfidf_bow = {
        'dataset' : DATASET,
        'name' : [name],
        'author_id' : AUTHOR_ID,
        'feature_extractor_ngram_range': [(3,4)],
        'feature_extractor_max_features': [1000],
        'feature_type': ['tfidf'],
        'feature_analyzer': ['char'],
        'samples': [1000],
        'special_char': [False],
        'word_length_dist': [False],
        'include_vocab_richness': [True],
        
        'svm_parameters' : svm_parameters,
        'lr_parameters': lr_parameters,
        'NaiveBayes_parameters': naiveBayes_parameters,
        'clf': clf,
        'ra' : ra,
        'ra_k' : ra_k,
        'ra_d' : ra_d,
        'ra_sentence_size' : ra_sentence_size,
        'ra_PCC_part_size' : ra_PCC_part_size,
        'ra_number_of_ra_inserts' : number_of_ra_inserts,
        'insert_cc' : insert_cc
    }

parameters_dependency = {
        'dataset' : DATASET,
        'name' : [name],
        'author_id' : AUTHOR_ID,
        'feature_type': ['dependency'],
        'samples': [1000],
        'special_char': [False],
        'word_length_dist': [False],
        'include_vocab_richness': [True],
        
        'svm_parameters' : svm_parameters,
        'lr_parameters': lr_parameters,
        'NaiveBayes_parameters': naiveBayes_parameters,
        'clf': clf,
        'ra' : ra,
        'ra_k' : ra_k,
        'ra_d' : ra_d,
        'ra_sentence_size' : ra_sentence_size,
        'ra_PCC_part_size' : ra_PCC_part_size,
        'ra_number_of_ra_inserts' : number_of_ra_inserts,
        'insert_cc' : insert_cc
    }

parameters_word_embeddings = {
        'dataset' : DATASET,
        'name' : [name],
        'author_id' : AUTHOR_ID,
        'feature_type': ['word_embeddings'],
        'samples': [870],
        'word_length_dist': [False],
        'include_vocab_richness': [True, False],
        
        'svm_parameters' : svm_parameters,
        'lr_parameters': lr_parameters,
        'NaiveBayes_parameters': naiveBayes_parameters,
        'clf': clf,
        'ra' : ra,
        'ra_k' : ra_k,
        'ra_d' : ra_d,
        'ra_sentence_size' : ra_sentence_size,
        'ra_PCC_part_size' : ra_PCC_part_size,
        'ra_number_of_ra_inserts' : number_of_ra_inserts,
    }

parameters_bert = {
        'dataset' : DATASET,
        'name' : [name],
        'author_id' : AUTHOR_ID,
        'feature_type': ['bert_m'],
        'samples': [870],
        'word_length_dist': [False],
        'include_vocab_richness': [True, False],
        
        'svm_parameters' : svm_parameters,
        'lr_parameters': lr_parameters,
        'NaiveBayes_parameters': naiveBayes_parameters,
        'clf': clf,
        'ra' : ra,
        'ra_k' : ra_k,
        'ra_d' : ra_d,
        'ra_sentence_size' : ra_sentence_size
    }

def base_experiment(parameters):

    nested_dicts = {k: parameters[k] for k in ['clf','svm_parameters', 'lr_parameters', 'NaiveBayes_parameters','dataset']}
    other_params = {k: parameters[k] for k in parameters if k not in nested_dicts}

    # Make sure all values are iterable
    for key, value in other_params.items():
        if not isinstance(value, (list, tuple)):  # Check if it's already iterable
            other_params[key] = [value]  # Convert single values to a list

    # Create combinations, filtering out those where d >= k
    combinations = []
    for v in itertools.product(*other_params.values()):
        comb = dict(zip(other_params, v))
        if comb['ra_d'] < comb['ra_k']:  # Add the check here
            combinations.append(comb)

    for combination in combinations:
        for k, v in nested_dicts.items():
            combination[k] = v

    return combinations

def pan20_ra2_SVM_tfidf(name):
    parameters_tfidf_bow['name'] = [name]
    parameters_tfidf_bow['distribution_plot'] = [True]
    return base_experiment(parameters_tfidf_bow)

def pan20_ra2_SVM_dependency(name):
    parameters_dependency['name'] = [name]
    parameters_dependency['distribution_plot'] = [True]
    return base_experiment(parameters_dependency)