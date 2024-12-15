import itertools

from settings.static_values import EXPECTED_DATASETS_FOLDER
from dataset_changes.pan20_create_baseline_1 import DATASET_CREATE_PATH

name = 'default'

DATASET = {
    'dataset': 'rolling-attribution-1',
    'dataset_train_path_pair': EXPECTED_DATASETS_FOLDER + DATASET_CREATE_PATH + "/pan20-train-pairs-AUTHOR.jsonl",
    'dataset_train_path_truth': EXPECTED_DATASETS_FOLDER + DATASET_CREATE_PATH + "/pan20-train-truth-AUTHOR.jsonl",
    'dataset_test_path_pair': EXPECTED_DATASETS_FOLDER + DATASET_CREATE_PATH + "/pan20-test-pairs-AUTHOR.jsonl",
    'dataset_test_path_truth': EXPECTED_DATASETS_FOLDER + DATASET_CREATE_PATH + "/pan20-test-truth-AUTHOR.jsonl",
    'pcc_train_samples': EXPECTED_DATASETS_FOLDER + "/pan20-created/pan20-train-all-different-authors.jsonl",
    'pcc_test_samples': EXPECTED_DATASETS_FOLDER + "pan20-created/pan20-test-all-different-authors.jsonl"
}

svm_parameters = {
    'svm_c': [10],
    'svm_degree': [1],
    'svm_kernel': ['poly']
}

lr_parameters = {
            'lr_c': [1],
            'lr_penalty': ['elasticnet'],
            'lr_solver': ['saga'],
            'lr_l1_ratio': [0.4444],
            'lr_max_iter': [2000]
        }

naiveBayes_parameters = {
    'nb_alpha': [0],
    'nb_fit_prior': [0]
}

# AUTHOR PARAMS
# AUTHOR_ID = [2049660, 3107154, 4483094]
#     \item     1.0           2049660
#     \item     0.8999        3107154
#     \item     0.8099        4483094

# AUTHOR PARAMS
AUTHOR_ID = [2049660, 1648312, 1777261]
#     \item     1.0       2049660
#     \item     0.909     1648312
#     \item     0.8       1777261


def create_experiment(clf_type='SVM', k=2, d=1, X=20, Z=1, Y=1, 
                      feature_type='tfidf', feature_extractor_ngram_range=(3, 4), feature_extractor_max_features=1000,
                      feature_analyzer='char', samples=1000, special_char=False, word_length_dist=False, 
                      include_vocab_richness=True, name='default', AUTHOR_ID=2049660,insert_cc=True):
    ra_k = k
    ra_d = d 
    ra_sentence_size = X
    ra_PCC_part_size = Z 
    ra_number_of_ra_inserts = Y
    
    clf = {
        'SVM': [clf_type == 'SVM'],
        'LR': [clf_type == 'LR'],
        'NaiveBayes': [False]  # Assuming you're not using Naive Bayes here
    }
    
    parameters = {
        'dataset': DATASET,
        'name': [name],
        'author_id': [AUTHOR_ID],  # Use the current author_id
        'feature_type': [feature_type],
        'samples': [samples],
        'special_char': [special_char],
        'word_length_dist': [word_length_dist],
        'include_vocab_richness': [include_vocab_richness],
        'svm_parameters': svm_parameters,
        'lr_parameters': lr_parameters,
        'NaiveBayes_parameters': naiveBayes_parameters,
        'clf': clf,
        'ra': [True],  
        'ra_k': [ra_k],
        'ra_d': [ra_d],
        'ra_sentence_size': [ra_sentence_size],
        'ra_PCC_part_size': [ra_PCC_part_size],
        'ra_number_of_ra_inserts': [ra_number_of_ra_inserts],
        'insert_cc' : [insert_cc]
    }

    if feature_type == 'tfidf':
        parameters['feature_extractor_ngram_range'] = [feature_extractor_ngram_range]
        parameters['feature_extractor_max_features'] = [feature_extractor_max_features]
        parameters['feature_analyzer'] = [feature_analyzer]

    nested_dicts = {k: parameters[k] for k in ['clf', 'svm_parameters', 'lr_parameters', 'NaiveBayes_parameters', 'dataset']}
    other_params = {k: parameters[k] for k in parameters if k not in nested_dicts}
    combination = {}
    for k, v in nested_dicts.items():
        combination[k] = v
    for k, v in other_params.items():
        combination[k] = v[0]  # Extract the single value from the list


    return combination  # Return the list of configurations

