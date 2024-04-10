import itertools
import numpy as np

clf = {
        'SVM' : [True],
        'LR' : [True],
        'NaiveBayes' : [True]
    }

svm_parameters = {
            'svm_c': [0.1,0.05,1,5,10],
            'svm_degree': [1,2,3,4]
        }
lr_parameters = {
            'lr_c': [0.01, 0.1, 1, 10],
            'lr_penalty': ['elasticnet'],
            'lr_solver': ['saga'],
            'lr_l1_ratio': np.linspace(0, 1, 10),
            'lr_max_iter': [2000]
        }

naiveBayes_parameters = {
            'nb_alpha': [0.01, 0.1, 0.5, 1.0, 2.0],
            'nb_fit_prior': [True, False]
        }

# BASE PARAMETERS
ra = [True]
ra_k = [4,5]
ra_d = [1,2,3]
ra_sentence_size = [50]
samples = [100]

special_char = [True, False]
word_length_dist = [True, False]
include_vocab_richness = [True, False]

parameters_tfidf_bow = {
        'feature_extractor_ngram_range': [(1,1),(2,2),(3,3),(4,4),(5,5)],
        'feature_extractor_max_features': [1000],
        'feature_type': ['tfidf','BoW'],
        'feature_analyzer': ['word','char','char-wb'],
        'samples': samples,
        'special_char': special_char,
        'word_length_dist': word_length_dist,
        'include_vocab_richness': include_vocab_richness,
        
        'svm_parameters' : svm_parameters,
        'lr_parameters': lr_parameters,
        'NaiveBayes_parameters': naiveBayes_parameters,
        'clf': clf,
        'ra' : ra,
        'ra_k' : ra_k,
        'ra_d' : ra_d,
        'ra_sentence_size' : ra_sentence_size
    }

parameters_dependency = {
        'feature_type': ['dependency'],
        'samples': samples,
        'special_char': special_char,
        'word_length_dist': word_length_dist,
        'include_vocab_richness': include_vocab_richness,
        
        'svm_parameters' : svm_parameters,
        'lr_parameters': lr_parameters,
        'NaiveBayes_parameters': naiveBayes_parameters,
        'clf': clf,
        'ra' : ra,
        'ra_k' : ra_k,
        'ra_d' : ra_d,
        'ra_sentence_size' : ra_sentence_size
    }

parameters_word_embeddings = {
        'feature_type': ['word_embeddings'],
        'samples': samples,
        'word_length_dist': word_length_dist,
        'include_vocab_richness': include_vocab_richness,
        
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

def experiement_tfidf_bow_ra():
    return base_experiment(parameters_tfidf_bow)

def experiement_dependency_ra():
    return base_experiment(parameters_dependency)

def experiement_word_embeddings_ra():
    return base_experiment(parameters_word_embeddings)