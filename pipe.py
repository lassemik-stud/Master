# import pandas as pd
# import pickle
import time
import itertools
import warnings
from datetime import timedelta
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool#, cpu_count

from feature_extraction import TextPairFeatureExtractor
from preprocess import load_or_process_data #, save_data_to_pickle, load_data_from_pickle, data_exists
from evaluation import evaluations
#from experiments.base_experiment import experiement_tfidf_bow, experiement_word_embeddings, experiement_dependency#, experiment_bert
#from experiments.ra_experiment import experiement_tfidf_bow_ra, experiement_dependency_ra, experiement_word_embeddings_ra, experiment_bert_ra
# from experiments.experiment_troubleshooting import th_experiement_tfidf_bow_ra

from experiments.pan20_baseline_0 import experiement_tfidf_bow, experiement_dependency
from experiments.pan20_baseline_0_single_experiment import single_experiment

# from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)

from settings.logging import printLog

def run_pipeline_wrapper(args):
    return run_pipeline(*args)

def run_pipeline(pipeline:Pipeline, x_train, y_train, x_test, y_test, arg, classifier_name, pcc_test_param, raw_c_test):
    #feature_type = arg.get('feature_type')
    pipeline.fit(x_train, y_train)
    y_pred_proba = pipeline.predict_proba(x_test)
    evaluations(y_test, y_pred_proba[:, 1], arg, classifier_name, pcc_test_param, raw_c_test)

def svm(svm_combination, text_pair_feature_extractor_prepopulated, x_train, y_train, x_test, y_test, arg, pcc_test_param, raw_c_test):
    svm_c = svm_combination.get('svm_c')
    svm_degree = svm_combination.get('svm_degree')
    svm_kernel = svm_combination.get('svm_kernel')

    pipeline_svm = (Pipeline([
                    ('feature_extractor', text_pair_feature_extractor_prepopulated),
                    ('feature_selection',SelectKBest(chi2, k=100)),
                    ('scaler', StandardScaler()),
                    ('classifier', SVC(C=svm_c, kernel=svm_kernel, gamma='scale', degree=svm_degree, probability=True))
                ]))
    
    current_arg = arg.copy()
    svm_parameters = {
        'svm_c': svm_c,
        'svm_degree': svm_degree,
        'svm_kernel' : svm_kernel
    }
    current_arg.update({
        'svm_parameters' : svm_parameters,
        'lr_parameters': 0,
        'naive_bayes_parameters': 0,
    })
    run_pipeline(pipeline_svm, x_train, y_train, x_test, y_test, current_arg, 'SVM', pcc_test_param, raw_c_test)

def svm_wrapper(args):
    return svm(*args)

def lr(lr_combination, text_pair_feature_extractor_prepopulated, x_train, y_train, x_test, y_test, arg, pcc_test_param, raw_c_test):
    lr_c = lr_combination.get('lr_c')
    lr_penalty = lr_combination.get('lr_penalty')
    lr_solver = lr_combination.get('lr_solver')
    lr_l1_ratio = lr_combination.get('lr_l1_ratio')
    lr_max_iter = lr_combination.get('lr_max_iter')

    pipeline_lr=(Pipeline([
                    ('feature_extractor', text_pair_feature_extractor_prepopulated),
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
        'naive_bayes_parameters': 0,
    })
    run_pipeline(pipeline_lr, x_train, y_train, x_test, y_test, current_arg, 'LR', pcc_test_param, raw_c_test)

def lr_wrapper(args):
    return lr(*args)

def naive_bayes_wrapper(args):
    return naive_bayes(*args)

def naive_bayes(naive_bayes_combination, text_pair_feature_extractor_prepopulated, x_train, y_train, x_test, y_test, arg, pcc_test_param, raw_c_test):
    nb_alpha = naive_bayes_combination.get('nb_alpha')
    nb_fit_prior = naive_bayes_combination.get('nb_fit_prior')
    pipeline_nb = (Pipeline([
                    ('feature_extractor', text_pair_feature_extractor_prepopulated),
                    ('feature_selection',SelectKBest(chi2, k=100)),
                    ('scaler', MinMaxScaler()),
                    ('classifier', MultinomialNB(alpha=nb_alpha, fit_prior=nb_fit_prior))
    ]))

    current_arg = arg.copy()
    naive_bayes_parameters = {
        'nb_alpha': nb_alpha,
        'nb_fit_prior': nb_fit_prior
    }
    current_arg.update({
        'svm_parameters' : 0,
        'lr_parameters': 0,
        'NaiveBayes_parameters': naive_bayes_parameters,
    })

    run_pipeline(pipeline_nb, x_train, y_train, x_test, y_test, current_arg, 'NaiveBayes', pcc_test_param, raw_c_test)

def run_svm(clf_svm_flag, svm_combinations, text_pair_feature_extractor_prepopulated, x_train, y_train, x_test, y_test, arg, pcc_test_param, raw_c_test):
    if bool(clf_svm_flag[0]):
        printLog.debug(f'Running SVM with {len(svm_combinations)} combination(s)')
        with Pool() as p:
            p.map(svm_wrapper, [(svm_combination, text_pair_feature_extractor_prepopulated, x_train, y_train, x_test, y_test, arg, pcc_test_param,raw_c_test) for svm_combination in svm_combinations])

def run_lr(clf_lr_flag, lr_combinations, text_pair_feature_extractor_prepopulated, x_train, y_train, x_test, y_test, arg, pcc_test_param, raw_c_test):
    if bool(clf_lr_flag[0]):
        printLog.debug(f'Running LR with {len(lr_combinations)} combination(s)')
        with Pool() as p:
            p.map(lr_wrapper, [(lr_combination, text_pair_feature_extractor_prepopulated, x_train, y_train, x_test, y_test, arg, pcc_test_param, raw_c_test) for lr_combination in lr_combinations])

def run_nb(clf_nb_flag, naive_bayes_combinations, text_pair_feature_extractor_prepopulated, x_train, y_train, x_test, y_test, arg, pcc_test_param, raw_c_test):
    if bool(clf_nb_flag[0]):
        printLog.debug(f'Running Naive Bayes with {len(naive_bayes_combinations)} combination(s)')
        with Pool() as p:
            p.map(naive_bayes_wrapper, [(naive_bayes_combination, text_pair_feature_extractor_prepopulated, x_train, y_train, x_test, y_test, arg, pcc_test_param,raw_c_test) for naive_bayes_combination in naive_bayes_combinations])

def prepare_pipeline(arg):

    author_id = arg.get('author_id')

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
    text_pair_feature_extractor_prepopulated = TextPairFeatureExtractor(
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

    naive_bayes_parameters = arg.get('NaiveBayes_parameters')
    naive_bayes_combinations = [dict(zip(naive_bayes_parameters, v)) for v in itertools.product(*naive_bayes_parameters.values())] if naive_bayes_parameters else None

    # Rolling selection parameters
    ra = arg.get('ra')
      
    sentence_size = arg.get('ra_sentence_size') if ra else None
    k = arg.get('ra_k') if ra else None
    d = arg.get('ra_d') if ra else None
    pcc_rate = arg.get('ra_number_of_ra_inserts') if ra else None
    pcc_part_size = arg.get('ra_PCC_part_size') if ra else None
  
    x_train, y_train, x_test, y_test, raw_c_train, raw_c_test, pcc_train_params, pcc_test_param = load_or_process_data(cutoff=cutoff,sentence_size=sentence_size,k=k,d=d,arg=arg, author_id=author_id, dataset='baseline-0')
    printLog.debug(f'x_t: {len(x_train)}, y_t: {len(y_train)}, x_t: {len(x_test)}, y_t: {len(y_test)}, r_c_t: {len(raw_c_train)}, r_c_t: {len(raw_c_test)}')
    if ra: 
        printLog.debug(f'k: {k}, d: {d}, s_s: {sentence_size}, pcc_r: {pcc_rate}, pcc_part_size: {pcc_part_size}, author_id: {author_id}')
    # Initialize the pipelines
    # if clf_svm_flag:
    #     for svm_i, svm_combination in enumerate(svm_combinations):
    #         printLog.debug(f'Running SVM with combination {svm_i+1} of {len(svm_combinations)}')
    #         svm(svm_combination, text_pair_feature_extractor_prepopulated, x_train, y_train, x_test, y_test, arg, pcc_test_param, raw_c_test)

    #if clf_lr_flag:
    #    for lr_i, lr_combination in enumerate(lr_combinations):
    #        printLog.debug(f'Running LR with combination {lr_i+1} of {len(lr_combinations)}')
    #        lr(lr_combination, text_pair_feature_extractor_prepopulated, x_train, y_train, x_test, y_test, arg, pcc_test_param, raw_c_test)

    # if clf_nb_flag:
    #     for nb_i, naive_bayes_combination in enumerate(naive_bayes_combinations):
    #         printLog.debug(f'Running Naive Bayes with combination {nb_i+1} of {len(naive_bayes_combinations)}')
    #         naive_bayes(naive_bayes_combination, text_pair_feature_extractor_prepopulated, x_train, y_train, x_test, y_test, arg, pcc_test_param, raw_c_test)

    with ProcessPoolExecutor() as executor:
        executor.submit(run_svm, clf_svm_flag, svm_combinations, text_pair_feature_extractor_prepopulated, x_train, y_train, x_test, y_test, arg, pcc_test_param, raw_c_test)
        executor.submit(run_lr, clf_lr_flag, lr_combinations, text_pair_feature_extractor_prepopulated, x_train, y_train, x_test, y_test, arg, pcc_test_param, raw_c_test)
        executor.submit(run_nb, clf_nb_flag, naive_bayes_combinations, text_pair_feature_extractor_prepopulated, x_train, y_train, x_test, y_test, arg, pcc_test_param, raw_c_test)

def run_experiment(arguments, _type):
    durations = []
    printLog.debug(f'Running {_type}-experiment with {len(arguments)} combination(s)')
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

# _NAME = 'baseline-0-single-experiment-NB'
# single_experiment_arguments = single_experiment(_NAME)
# run_experiment(single_experiment_arguments, _NAME)

_NAME = 'baseline-0-dependency-experiment'
experiement_dependency_arguments = experiement_dependency(_NAME)
run_experiment(experiement_dependency_arguments, _NAME)

#th_experiement_tfidf_bow_ra = th_experiement_tfidf_bow_ra('experiment_prod_pan20')
#tfidf_arguments = experiement_tfidf_bow('experiment_prod_pan20-super')
#word_embeddings_arguments = experiement_word_embeddings('experiment_prod_pan20-super')
#dependency_arguments = experiement_dependency('experiment_prod_pan20-super')
#bert_arguments = experiemtn_bert_m()

##tfidf_ra_arguments = experiement_tfidf_bow_ra()
#word_embeddings_arguments_ra = experiement_word_embeddings_ra()
#dependency_arguments_ra = experiement_dependency_ra()
#bert_arguments_ra = experiment_bert_ra()

#run_experiment(th_experiement_tfidf_bow_ra, 'th_tfidf_ra')

#tfidf_arguments = experiement_tfidf_bow('b0-tfidf-experiment')

#run_experiment(tfidf_arguments, 'b0-tfidf-experiment')
#run_experiment(word_embeddings_arguments, 'word_embeddings')
#run_experiment(dependency_arguments, 'dependency')

# #run_experiment(bert_arguments_ra, 'bert-ra')
# run_experiment(tfidf_ra_arguments, 'tfidf-ra')
# run_experiment(word_embeddings_arguments_ra, 'word_embeddings-ra')
# run_experiment(dependency_arguments_ra, 'dependency-ra')

# #run_experiment(word_embeddings_arguments, 'word_embeddings')
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
        # pipeline.fit(x_train, y_train)
        # y_pred_proba = pipeline.predict_proba(x_test)
        # y_pred = pipeline.predict(x_test)
        # #    save_data_to_pickle(y_pred, path)
        # #   printLog.debug('Classification data saved to pickle files')
        
        # evaluations(y_test, y_pred_proba[:, 1], arg)
        # #distribution_plot(y_test, y_pred_proba[:, 1], arg)
