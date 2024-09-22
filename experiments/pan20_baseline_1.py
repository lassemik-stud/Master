import itertools
import numpy as np

from settings.static_values import EXPECTED_DATASETS_FOLDER
from dataset_changes.pan20_create_baseline_1 import DATASET_CREATE_PATH

name = 'default'

DATASET = {
    'dataset' : 'baseline-1',
    'dataset_train_path_pair' : EXPECTED_DATASETS_FOLDER + DATASET_CREATE_PATH + "/pan20-train-pairs-AUTHOR.jsonl",
    'dataset_train_path_truth': EXPECTED_DATASETS_FOLDER + DATASET_CREATE_PATH + "/pan20-train-truth-AUTHOR.jsonl",
    'dataset_test_path_pair' : EXPECTED_DATASETS_FOLDER + DATASET_CREATE_PATH + "/pan20-test-pairs-AUTHOR.jsonl",
    'dataset_test_path_truth' : EXPECTED_DATASETS_FOLDER + DATASET_CREATE_PATH + "/pan20-test-truth-AUTHOR.jsonl"
}

# base experiment that does something

clf = {
        'SVM' : [True],
        'LR' : [True],
        'NaiveBayes' : [True]
    }

svm_parameters = {
            'svm_c': [0.01, 0.1, 1, 10],
            'svm_degree': [1,2,3],
            'svm_kernel': ['linear', 'poly', 'rbf', 'sigmoid']
        }
lr_parameters = {
            'lr_c': [0.01, 0.1, 1, 10],
            'lr_penalty': ['elasticnet'],
            'lr_solver': ['saga'],
            'lr_l1_ratio': np.linspace(0, 1, 10),
            'lr_max_iter': [2000]
        }

naiveBayes_parameters = {
            'nb_alpha': [0.01, 0.1, 0.5, 1.0, 10],
            'nb_fit_prior': [True, False]
        }

with open('../dataset_changes/pan20_similar_authors.txt', 'r', encoding='utf-8') as file:
    author_id = [line.strip() for line in file if line.strip()]

AUTHOR_ID = [str(id) for id in author_id]

# BASE PARAMETERS
ra = [False]
ra_k = [0]                  # Window size
ra_d = [0]                # Overlap size
ra_sentence_size = [0]     # Size of part used to split up the text. This could be a paragraph size. Size is in number of sentences. 
samples = [100]             # Max number of samples used in train and test
ra_PCC_part_size = [1]      # number of parts of sentence size inserted into the UT
number_of_ra_inserts = [1]

special_char = [True]
word_length_dist = [False]
include_vocab_richness = [False]

parameters_tfidf_bow = {
        'dataset' : DATASET,
        'name' : [name],
        'author_id' : {AUTHOR_ID},
        'feature_extractor_ngram_range': [(1,1),(2,2),(3,3),(4,4),(5,5),(1,2),(2,3),(3,4),(4,5)],
        'feature_extractor_max_features': [1000],
        'feature_type': ['tfidf','BoW'],
        'feature_analyzer': ['word', 'char', 'char_wb'],
        'samples': [870],
        'special_char': [True, False],
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

parameters_dependency = {
        'dataset' : DATASET,
        'name' : [name],
        'author_id' : {AUTHOR_ID},
        'feature_type': ['dependency'],
        'samples': [870],
        'special_char': [True, False],
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

parameters_word_embeddings = {
        'dataset' : DATASET,
        'name' : [name],
        'author_id' : {AUTHOR_ID},
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
        'author_id' : {AUTHOR_ID},
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
    nested_dicts = {k: parameters[k] for k in ['clf','svm_parameters', 'lr_parameters', 'NaiveBayes_parameters']}
    other_params = {k: parameters[k] for k in parameters if k not in nested_dicts}

    combinations = [dict(zip(other_params, v)) for v in itertools.product(*other_params.values())]

    for combination in combinations:
        for k, v in nested_dicts.items():
            combination[k] = v

    return combinations

def experiment_bert(name):
    parameters_bert['name'] = [name]
    return base_experiment(parameters_bert)

def experiement_tfidf_bow(name):
    parameters_tfidf_bow['name'] = [name]
    return base_experiment(parameters_tfidf_bow)

def experiement_dependency(name):
    parameters_dependency['name'] = [name]
    return base_experiment(parameters_dependency)

def experiement_word_embeddings(name):
    parameters_word_embeddings['name'] = [name]
    return base_experiment(parameters_word_embeddings)
