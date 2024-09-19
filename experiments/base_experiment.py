import itertools
import numpy as np

name = 'default'

# base experiment that 

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

# AUTHOR PARAMS 1000555, 1004266, 1044982, 1046661, 1059049, 1060901, 1067919 #,
author_id = [1102473, 1124370, 1134135, 1144748, 1207016, 1236246, 1254171, 1259527, 134389, 1354544, 1384142, 139401, 1431943, 146806, 1492084, 150067, 1555294, 1576308, 159540, 1597786, 1641001, 1648312, 1652711, 1655407, 1716956, 1777261, 1783027, 1796268, 1810674, 182830, 1862439, 1869762, 187436, 1952016, 1956189, 1957052, 2002255, 2007348, 2031530, 204149, 2049660, 2129042, 2135508, 214030, 2213298, 2299844, 2352342, 2376938, 240162, 2456602, 2469390, 25619, 2664779, 2669603, 267735, 2688002, 270548, 2738227, 2762778, 283936, 284145, 296467, 298653, 3090681, 3107154, 31351, 32276, 3231678, 324872, 3561385, 357927, 3628045, 3667168, 3669238, 3735343, 3993743, 404703, 429953, 4339208, 4373288, 437416, 4415171, 442738, 44720, 4483094, 4787616, 480321, 4865253, 526713, 5430304, 547570, 55318, 561615, 56264, 578300, 607817, 610733, 627559, 646233, 649516, 70311, 709114, 744563, 74824, 763713, 76380, 80018, 806976, 870118, 882056, 900596, 909661, 913162, 9154517, 920809, 951853, 974478, 1000555, 1004266, 1044982, 1046661, 1059049, 1060901, 1067919, 1102473, 1124370, 1134135, 1144748, 1207016, 1236246, 1254171, 1259527, 134389, 1354544, 1384142, 139401, 1431943, 146806, 1492084, 150067, 1555294, 1576308, 159540, 1597786, 1641001, 1648312, 1652711, 1655407, 1716956, 1777261, 1783027, 1796268, 1810674, 182830, 1862439, 1869762, 187436, 1952016, 1956189, 1957052, 2002255, 2007348, 2031530, 204149, 2049660, 2129042, 2135508, 214030, 2213298, 2299844, 2352342, 2376938, 240162, 2456602, 2469390, 25619, 2664779, 2669603, 267735, 2688002, 270548, 2738227, 2762778, 283936, 284145, 296467, 298653, 3090681, 3107154, 31351, 32276, 3231678, 324872, 3561385, 357927, 3628045, 3667168, 3669238, 3735343, 3993743, 404703, 429953, 4339208, 4373288, 437416, 4415171, 442738, 44720, 4483094, 4787616, 480321, 4865253, 526713, 5430304, 547570, 55318, 561615, 56264, 578300, 607817, 610733, 627559, 646233, 649516, 70311, 709114, 744563, 74824, 763713, 76380, 80018, 806976, 870118, 882056, 900596, 909661, 913162, 9154517, 920809, 951853, 974478, 1000555, 1004266, 1044982, 1046661, 1059049, 1060901, 1067919, 1102473, 1124370, 1134135, 1144748, 1207016, 1236246, 1254171, 1259527, 134389, 1354544, 1384142, 139401, 1431943, 146806, 1492084, 150067, 1555294, 1576308, 159540, 1597786, 1641001, 1648312, 1652711, 1655407, 1716956, 1777261, 1783027, 1796268, 1810674, 182830, 1862439, 1869762, 187436, 1952016, 1956189, 1957052, 2002255, 2007348, 2031530, 204149, 2049660, 2129042, 2135508, 214030, 2213298, 2299844, 2352342, 2376938, 240162, 2456602, 2469390, 25619, 2664779, 2669603, 267735, 2688002, 270548, 2738227, 2762778, 283936, 284145, 296467, 298653, 3090681, 3107154, 31351, 32276, 3231678, 324872, 3561385, 357927, 3628045, 3667168, 3669238, 3735343, 3993743, 404703, 429953, 4339208, 4373288, 437416, 4415171, 442738, 44720, 4483094, 4787616, 480321, 4865253, 526713, 5430304, 547570, 55318, 561615, 56264, 578300, 607817, 610733, 627559, 646233, 649516, 70311, 709114, 744563, 74824, 763713, 76380, 80018, 806976, 870118, 882056, 900596, 909661, 913162, 9154517, 920809, 951853, 974478, 1000555, 1004266, 1044982, 1046661, 1059049, 1060901, 1067919, 1102473, 1124370, 1134135, 1144748, 1207016, 1236246, 1254171, 1259527, 134389, 1354544, 1384142, 139401, 1431943, 146806, 1492084, 150067, 1555294, 1576308, 159540, 1597786, 1641001, 1648312, 1652711, 1655407, 1716956, 1777261, 1783027, 1796268, 1810674, 182830, 1862439, 1869762, 187436, 1952016, 1956189, 1957052, 2002255, 2007348, 2031530, 204149, 2049660, 2129042, 2135508, 214030, 2213298, 2299844, 2352342, 2376938, 240162, 2456602, 2469390, 25619, 2664779, 2669603, 267735, 2688002, 270548, 2738227, 2762778, 283936, 284145, 296467, 298653, 3090681, 3107154, 31351, 32276, 3231678, 324872, 3561385, 357927, 3628045, 3667168, 3669238, 3735343, 3993743, 404703, 429953, 4339208, 4373288, 437416, 4415171, 442738, 44720, 4483094, 4787616, 480321, 4865253, 526713, 5430304, 547570, 55318, 561615, 56264, 578300, 607817, 610733, 627559, 646233, 649516, 70311, 709114, 744563, 74824, 763713, 76380, 80018, 806976, 870118, 882056, 900596, 909661, 913162, 9154517, 920809, 951853, 974478]
author_id = [str(id) for id in author_id]

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
        'name' : [name],
        'author_id' : author_id,
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
        'name' : [name],
        'author_id' : author_id,
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
