import itertools
import numpy as np

from settings.static_values import EXPECTED_DATASETS_FOLDER

name = 'default'

DATASET = {
    'dataset' : 'pan23',
    'dataset_train_path_pair' : EXPECTED_DATASETS_FOLDER + "pan23_authorship-verification-training-dataset/pairs.jsonl",
    'dataset_train_path_truth': EXPECTED_DATASETS_FOLDER + "pan23_authorship-verification-training-dataset/truth.jsonl",
    'dataset_test_path_pair' : EXPECTED_DATASETS_FOLDER + "pan23-authorship-verification-test-dataset/pairs.jsonl",
    'dataset_test_path_truth' : EXPECTED_DATASETS_FOLDER + "pan23-authorship-verification-test-dataset/truth.jsonl"
}

clf = {
        'SVM' : [True],
        'LR' : [False],
        'NaiveBayes' : [False]
    }

svm_parameters = {
            'svm_c': [0.1],
            'svm_degree': [3],
            'svm_kernel': ['poly']
        }

lr_parameters = {
            'lr_c': [0.01],
            'lr_penalty': ['elasticnet'],
            'lr_solver': ['saga'],
            'lr_l1_ratio': [0.0],
            'lr_max_iter': [2000]
        }

naiveBayes_parameters = {
            'nb_alpha': [10],
            'nb_fit_prior': [False]
        }

author_id = 0

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
        'author_id' : author_id,
        'feature_extractor_ngram_range': [(4,5)],
        'feature_extractor_max_features': [1000],
        'feature_type': ['tfidf'],
        'feature_analyzer': ['char'],
        'samples': [8836],
        'special_char': [False],
        'word_length_dist': [False],
        'include_vocab_richness': [False],
        
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
        'author_id' : author_id,
        'feature_type': ['dependency'],
        'samples': [870],
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
    }

parameters_word_embeddings = {
        'dataset' : DATASET,
        'name' : [name],
        'author_id' : author_id,
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
        'author_id' : author_id,
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

    combinations = [dict(zip(other_params, v)) for v in itertools.product(*other_params.values())]

    for combination in combinations:
        for k, v in nested_dicts.items():
            combination[k] = v

    return combinations

def pan23_single_experiment_tfidf(name):
    parameters_tfidf_bow['name'] = [name]
    parameters_tfidf_bow['distribution_plot'] = [True]
    return base_experiment(parameters_tfidf_bow)

def pan23_single_experiment_dependency(name):
    parameters_dependency['name'] = [name]
    parameters_dependency['distribution_plot'] = [True]
    return base_experiment(parameters_dependency)