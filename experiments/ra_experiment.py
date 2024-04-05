import itertools
import numpy as np

clf = {
        'SVM' : [True],
        'LR' : [True],
        'NaiveBayes' : [True]
    }

svm_parameters = {
            'svm_c': [0.001], #, 0.1, 1, 10],
            'svm_degree': [1] #,2,3]
        }
lr_parameters = {
            'lr_c': [0.01],# 0.1, 1, 10],
            'lr_penalty': ['elasticnet'],
            'lr_solver': ['saga'],
            'lr_l1_ratio': [0], #np.linspace(0, 1, 10),
            'lr_max_iter': [2000]
        }

naiveBayes_parameters = {
            'nb_alpha': [0.01],# 0.1, 0.5, 1.0, 2.0],
            'nb_fit_prior': [True] #, False]
        }

# BASE PARAMETERS
ra = [True]
ra_k = [1,2,3,4,5]                  # Window size
ra_d = [0,1,2,3,4]                # Overlap size
ra_sentence_size = [30,50,80]     # Size of part used to split up the text. This could be a paragraph size. Size is in number of sentences. 
samples = [100]             # Max number of samples used in train and test
ra_PCC_part_size = [1,2,3,4]      # number of parts of sentence size inserted into the UT


special_char = [True, False]
word_length_dist = [True, False]
include_vocab_richness = [True, False]

parameters_tfidf_bow = {
        'feature_extractor_ngram_range': [(4,4),(5,5)],
        'feature_extractor_max_features': [1000],
        'feature_type': ['tfidf','BoW'],
        'feature_analyzer': ['word','char','char_wb'],
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
        'ra_sentence_size' : ra_sentence_size,
        'ra_PCC_part_size' : ra_PCC_part_size
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
        'ra_sentence_size' : ra_sentence_size,
        'ra_PCC_part_size' : ra_PCC_part_size
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
        'ra_sentence_size' : ra_sentence_size,
        'ra_PCC_part_size' : ra_PCC_part_size
    }

parameters_bert = {
        'feature_type': ['bert_m'],
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
        'ra_sentence_size' : ra_sentence_size,
        'ra_PCC_part_size' : ra_PCC_part_size
    }

def base_experiment(parameters):
    nested_dicts = {k: parameters[k] for k in ['clf','svm_parameters', 'lr_parameters', 'NaiveBayes_parameters']}
    other_params = {k: parameters[k] for k in parameters if k not in nested_dicts}

    combinations = [dict(zip(other_params, v)) for v in itertools.product(*other_params.values())]

    valid_combinations = []
    for combination in combinations:
        if combination['ra_d'] >= combination['ra_k']:
            continue  
        for k, v in nested_dicts.items():
            combination[k] = v
        valid_combinations.append(combination)

    return valid_combinations

def experiement_tfidf_bow_ra():
    return base_experiment(parameters_tfidf_bow)

def experiement_dependency_ra():
    return base_experiment(parameters_dependency)

def experiement_word_embeddings_ra():
    return base_experiment(parameters_word_embeddings)

def experiment_bert_ra():
    return base_experiment(parameters_bert)