from sklearn.pipeline import Pipeline

from sklearn.metrics import classification_report
import pandas as pd
import pickle
import time
from datetime import timedelta
from concurrent.futures import ProcessPoolExecutor

from sklearn.preprocessing import MinMaxScaler

import itertools
from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, chi2


from settings.logging import printLog
from multiprocessing import Pool, cpu_count


import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)

from feature_extraction import TextPairFeatureExtractor
from preprocess import load_or_process_data, save_data_to_pickle, load_data_from_pickle, data_exists
from evaluation import evaluations
from experiments.base_experiment import experiement_tfidf_bow, experiement_word_embeddings, experiement_dependency
from experiments.ra_experiment import experiement_tfidf_bow_ra, experiement_dependency_ra, experiement_word_embeddings_ra, experiment_bert_ra

def run_pipeline_wrapper(args):
    return run_pipeline(*args)

def run_pipeline(PCC_Pipeline:Pipeline, x_train, y_train, x_test, y_test, arg, classifier_name, PCC_test_params, raw_c_test):
    feature_type = arg.get('feature_type')
    PCC_Pipeline.fit(x_train, y_train)
    y_pred_proba = PCC_Pipeline.predict_proba(x_test)
    evaluations(y_test, y_pred_proba[:, 1], arg, classifier_name, PCC_test_params, raw_c_test)

def svm(svm_combination, TextPairFeatureExtractorPrepopulated, x_train, y_train, x_test, y_test, arg, PCC_test_params, raw_c_test):
    svm_c = svm_combination.get('svm_c')
    svm_degree = svm_combination.get('svm_degree')

    PCC_Pipeline_SVM = (Pipeline([
                    ('feature_extractor', TextPairFeatureExtractorPrepopulated),
                    ('feature_selection',SelectKBest(chi2, k=100)),
                    ('scaler', StandardScaler()),
                    ('classifier', SVC(C=svm_c, kernel='sigmoid', gamma='scale', degree=svm_degree, probability=True))
                ]))
    
    current_arg = arg.copy()
    svm_parameters = {
        'svm_c': svm_c,
        'svm_degree': svm_degree
    }
    current_arg.update({
        'svm_parameters' : svm_parameters,
        'lr_parameters': 0,
        'NaiveBayes_parameters': 0,
    })
    run_pipeline(PCC_Pipeline_SVM, x_train, y_train, x_test, y_test, current_arg, 'SVM', PCC_test_params, raw_c_test)

def svm_wrapper(args):
    return svm(*args)

def lr(lr_combination, TextPairFeatureExtractorPrepopulated, x_train, y_train, x_test, y_test, arg, PCC_test_params, raw_c_test):
    lr_c = lr_combination.get('lr_c')
    lr_penalty = lr_combination.get('lr_penalty')
    lr_solver = lr_combination.get('lr_solver')
    lr_l1_ratio = lr_combination.get('lr_l1_ratio')
    lr_max_iter = lr_combination.get('lr_max_iter')

    PCC_Pipeline_LR=(Pipeline([
                    ('feature_extractor', TextPairFeatureExtractorPrepopulated),
                    ('feature_selection',SelectKBest(chi2, k=100)),
                    ('scaler', StandardScaler()),
                    ('classifier', LogisticRegression(C=lr_c, penalty=lr_penalty, solver=lr_solver, l1_ratio=lr_l1_ratio, max_iter=lr_max_iter))
                ]))

    current_arg = arg.copy()
    lr_parameters = {
        'lr_c': lr_c,
        'lr_penalty': lr_penalty,
        'lr_solver': lr_solver,
        'lr_l1_ratio': lr_l1_ratio,
        'lr_max_iter': lr_max_iter
    }
    current_arg.update({
        'svm_parameters' : 0,
        'lr_parameters': lr_parameters,
        'NaiveBayes_parameters': 0,
    })
    run_pipeline(PCC_Pipeline_LR, x_train, y_train, x_test, y_test, current_arg, 'LR', PCC_test_params)


def lr_wrapper(args):
    return lr(*args)

def naiveBayes_wrapper(args):
    return naiveBayes(*args)

def naiveBayes(NaiveBayes_combination, TextPairFeatureExtractorPrepopulated, x_train, y_train, x_test, y_test, arg, PCC_test_params, raw_c_test):
    nb_alpha = NaiveBayes_combination.get('nb_alpha')
    nb_fit_prior = NaiveBayes_combination.get('nb_fit_prior')
    PCC_Pipeline_NB = (Pipeline([
                    ('feature_extractor', TextPairFeatureExtractorPrepopulated),
                    ('feature_selection',SelectKBest(chi2, k=100)),
                    ('scaler', MinMaxScaler()),
                    ('classifier', MultinomialNB(alpha=nb_alpha, fit_prior=nb_fit_prior))
    ]))

    current_arg = arg.copy()
    naiveBayes_parameters = {
        'nb_alpha': nb_alpha,
        'nb_fit_prior': nb_fit_prior
    }
    current_arg.update({
        'svm_parameters' : 0,
        'lr_parameters': 0,
        'NaiveBayes_parameters': naiveBayes_parameters,
    })

    run_pipeline(PCC_Pipeline_NB, x_train, y_train, x_test, y_test, current_arg, 'NaiveBayes', PCC_test_params)

def run_svm(clf_svm_flag, svm_combinations, TextPairFeatureExtractorPrepopulated, x_train, y_train, x_test, y_test, arg, PCC_test_params, raw_c_test):
    if clf_svm_flag:
        printLog.debug(f'Running SVM with {len(svm_combinations)} combinations')
        with Pool() as p:
            p.map(svm_wrapper, [(svm_combination, TextPairFeatureExtractorPrepopulated, x_train, y_train, x_test, y_test, arg, PCC_test_params,raw_c_test) for svm_combination in svm_combinations])

def run_lr(clf_lr_flag, lr_combinations, TextPairFeatureExtractorPrepopulated, x_train, y_train, x_test, y_test, arg, PCC_test_params, raw_c_test):
    if clf_lr_flag:
        printLog.debug(f'Running LR with {len(lr_combinations)} combinations')
        with Pool() as p:
            p.map(lr_wrapper, [(lr_combination, TextPairFeatureExtractorPrepopulated, x_train, y_train, x_test, y_test, arg, PCC_test_params, raw_c_test) for lr_combination in lr_combinations])

def run_nb(clf_nb_flag, NaiveBayes_combinations, TextPairFeatureExtractorPrepopulated, x_train, y_train, x_test, y_test, arg, PCC_test_params, raw_c_test):
    if clf_nb_flag:
        printLog.debug(f'Running Naive Bayes with {len(NaiveBayes_combinations)} combinations')
        with Pool() as p:
            p.map(naiveBayes_wrapper, [(NaiveBayes_combination, TextPairFeatureExtractorPrepopulated, x_train, y_train, x_test, y_test, arg, PCC_test_params,raw_c_test) for NaiveBayes_combination in NaiveBayes_combinations])


def prepare_pipeline(arg):
    # Overall parameters
    cutoff = arg.get('samples')

    # Feature extraction parameters
    feature_extractor_ngram_range = arg.get('feature_extractor_ngram_range')
    feature_extractor_max_features = arg.get('feature_extractor_max_features')
    feature_type = arg.get('feature_type')
    feature_analyzer = arg.get('feature_analyzer')
    special_char = arg.get('special_char')
    word_length_dist = arg.get('word_length_dist')
    include_vocab_richness = arg.get('include_vocab_richness')
    TextPairFeatureExtractorPrepopulated = TextPairFeatureExtractor(
                                    ngram_range=feature_extractor_ngram_range,
                                    max_features=feature_extractor_max_features,
                                    feature_type=feature_type,
                                    analyzer=feature_analyzer,
                                    special_chars=special_char,
                                    word_length_dist=word_length_dist,
                                    include_vocab_richness=include_vocab_richness)

    # Classifier parameters
    clfs = arg.get('clf')
    clf_svm_flag = clfs.get('SVM')
    clf_lr_flag = clfs.get('LR')
    clf_nb_flag = clfs.get('NaiveBayes')

    svm_parameters = arg.get('svm_parameters')

    svm_combinations = [dict(zip(svm_parameters, v)) for v in itertools.product(*svm_parameters.values())] if svm_parameters else None
    
    lr_parameters = arg.get('lr_parameters')
    lr_combinations = [dict(zip(lr_parameters, v)) for v in itertools.product(*lr_parameters.values())] if lr_parameters else None

    NaiveBayes_parameters = arg.get('NaiveBayes_parameters')
    NaiveBayes_combinations = [dict(zip(NaiveBayes_parameters, v)) for v in itertools.product(*NaiveBayes_parameters.values())] if NaiveBayes_parameters else None

    # Rolling selection parameters
    ra = arg.get('ra')
        
    sentence_size = arg.get('ra_sentence_size') if ra else None
    k = arg.get('ra_k') if ra else None
    d = arg.get('ra_d') if ra else None

    x_train, y_train, x_test, y_test, raw_c_train, raw_c_test, PCC_train_params, PCC_test_params = load_or_process_data(cutoff=cutoff,sentence_size=sentence_size,k=k,d=d,arg=arg)
    printLog.debug(f'sizes: x_train: {len(x_train)}, y_train: {len(y_train)}, x_test: {len(x_test)}, y_test: {len(y_test)}, raw_c_train: {len(raw_c_train)}, raw_c_train: {len(raw_c_test)}')

    # Initialize the pipelines
    if clf_svm_flag:
        for svm_i, svm_combination in enumerate(svm_combinations):
            printLog.debug(f'Running SVM with combination {svm_i+1} of {len(svm_combinations)}')
            svm(svm_combination, TextPairFeatureExtractorPrepopulated, x_train, y_train, x_test, y_test, arg, PCC_test_params, raw_c_test)

    # with ProcessPoolExecutor() as executor:
    #     executor.submit(run_svm, clf_svm_flag, svm_combinations, TextPairFeatureExtractorPrepopulated, x_train, y_train, x_test, y_test, arg, PCC_test_params, raw_c_test)
    #     executor.submit(run_lr, clf_lr_flag, lr_combinations, TextPairFeatureExtractorPrepopulated, x_train, y_train, x_test, y_test, arg, PCC_test_params, raw_c_test)
    #     executor.submit(run_nb, clf_nb_flag, NaiveBayes_combinations, TextPairFeatureExtractorPrepopulated, x_train, y_train, x_test, y_test, arg, PCC_test_params, raw_c_test)

def run_experiment(arguments, _type):
    durations = []
    printLog.debug(f'Running {_type}-experiment with {len(arguments)} combinations')
    for i, argument in enumerate(arguments):
        start_time = time.time()
        
        printLog.info(f'{100*(i+1)/len(arguments):.2f} % --> Running {_type}-experiment {i+1} of {len(arguments)}')
        prepare_pipeline(argument)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        durations.append(elapsed_time)
        
        avg_time = sum(durations) / len(durations)
        remaining_experiments = len(arguments) - (i + 1)
        estimated_time_left = avg_time * remaining_experiments
        
        # Convert estimated time left to clock format
        eta = str(timedelta(seconds=int(estimated_time_left)))
        
        printLog.info(f'Experiment {i+1} took {elapsed_time:.2f} seconds. Estimated time left: {eta} (H:M:S).')
        
tfidf_arguments = experiement_tfidf_bow()
word_embeddings_arguments = experiement_word_embeddings()
dependency_arguments = experiement_dependency()

tfidf_ra_arguments = experiement_tfidf_bow_ra()
word_embeddings_arguments_ra = experiement_word_embeddings_ra()
dependency_arguments_ra = experiement_dependency_ra()
bert_arguments_ra = experiment_bert_ra()

# run_experiment(tfidf_arguments, 'tfidf')
# run_experiment(word_embeddings_arguments, 'word_embeddings')
# run_experiment(dependency_arguments, 'dependency')

run_experiment(bert_arguments_ra, 'bert-ra')
run_experiment(tfidf_ra_arguments, 'tfidf-ra')
run_experiment(word_embeddings_arguments_ra, 'word_embeddings-ra')
run_experiment(dependency_arguments_ra, 'dependency-ra')

#run_experiment(word_embeddings_arguments, 'word_embeddings')
#run_experiment(dependency_arguments, 'dependency')

#for arg in arguments:
    
    #prepare_pipeline(arg)
 #   exit()
    
        # # Preload the data if it has been processed before
        # #root_path = '../pre_data/'
        # #path = f'{root_path}-svc_c-{svc_c}-{svc_degree}-{str(cutoff)}-{str(sentence_size)}-{k}-{d}-ft-{feature_type}-fa{feature_analyzer}-ngram-{str(feature_extractor_ngram_range[0])}-{str(feature_extractor_ngram_range[1])}-{feature_extractor_max_features}-y_pred.pkl'
        # #if data_exists(path):
        # #    y_pred = load_data_from_pickle(path)
        # #    printLog.debug('Classification data loaded from pickle files')
        # #else:
        # PCC_Pipeline.fit(x_train, y_train)
        # y_pred_proba = PCC_Pipeline.predict_proba(x_test)
        # y_pred = PCC_Pipeline.predict(x_test)
        # #    save_data_to_pickle(y_pred, path)
        # #   printLog.debug('Classification data saved to pickle files')
        
        # evaluations(y_test, y_pred_proba[:, 1], arg)
        # #distribution_plot(y_test, y_pred_proba[:, 1], arg)
