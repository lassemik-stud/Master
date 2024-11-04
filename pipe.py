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
from evaluation import evaluations, get_best_auroc
#from experiments.base_experiment import experiement_tfidf_bow, experiement_word_embeddings, experiement_dependency#, experiment_bert
#from experiments.ra_experiment import experiement_tfidf_bow_ra, experiement_dependency_ra, experiement_word_embeddings_ra, experiment_bert_ra
# from experiments.experiment_troubleshooting import th_experiement_tfidf_bow_ra


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
    if (clf_svm_flag):
        printLog.debug(f'Running SVM with {len(svm_combinations)} combination(s)')
        with Pool() as p:
            p.map(svm_wrapper, [(svm_combination, text_pair_feature_extractor_prepopulated, x_train, y_train, x_test, y_test, arg, pcc_test_param,raw_c_test) for svm_combination in svm_combinations])

def run_lr(clf_lr_flag, lr_combinations, text_pair_feature_extractor_prepopulated, x_train, y_train, x_test, y_test, arg, pcc_test_param, raw_c_test):
    if (clf_lr_flag):
        printLog.debug(f'Running LR with {len(lr_combinations)} combination(s)')
        with Pool() as p:
            p.map(lr_wrapper, [(lr_combination, text_pair_feature_extractor_prepopulated, x_train, y_train, x_test, y_test, arg, pcc_test_param, raw_c_test) for lr_combination in lr_combinations])

def run_nb(clf_nb_flag, naive_bayes_combinations, text_pair_feature_extractor_prepopulated, x_train, y_train, x_test, y_test, arg, pcc_test_param, raw_c_test):
    if (clf_nb_flag):
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
    clf_svm_flag = bool(clfs.get('SVM')[0])
    clf_lr_flag = (clfs.get('LR')[0])
    clf_nb_flag = (clfs.get('NaiveBayes')[0])

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

    dataset = arg.get('dataset')
    selected_dataset = dataset.get('dataset')
  
    x_train, y_train, x_test, y_test, raw_c_train, raw_c_test, pcc_train_params, pcc_test_param = load_or_process_data(cutoff=cutoff,sentence_size=sentence_size,k=k,d=d,arg=arg, author_id=author_id, dataset=selected_dataset)
    printLog.debug(f'x_t: {len(x_train)}, y_t: {len(y_train)}, x_t: {len(x_test)}, y_t: {len(y_test)}, r_c_t: {len(raw_c_train)}, r_c_t: {len(raw_c_test)}')
    if ra: 
        printLog.debug(f'k: {k}, d: {d}, s_s: {sentence_size}, pcc_r: {pcc_rate}, pcc_part_size: {pcc_part_size}, author_id: {author_id}')
    # Initialize the pipelines
    if (clf_svm_flag):
        for svm_i, svm_combination in enumerate(svm_combinations):
            printLog.debug(f'Running SVM with combination {svm_i+1} of {len(svm_combinations)}')
            svm(svm_combination, text_pair_feature_extractor_prepopulated, x_train, y_train, x_test, y_test, arg, pcc_test_param, raw_c_test)

    if (clf_lr_flag):
       for lr_i, lr_combination in enumerate(lr_combinations):
           printLog.debug(f'Running LR with combination {lr_i+1} of {len(lr_combinations)}')
           lr(lr_combination, text_pair_feature_extractor_prepopulated, x_train, y_train, x_test, y_test, arg, pcc_test_param, raw_c_test)

    if (clf_nb_flag):
        for nb_i, naive_bayes_combination in enumerate(naive_bayes_combinations):
            printLog.debug(f'Running Naive Bayes with combination {nb_i+1} of {len(naive_bayes_combinations)}')
            naive_bayes(naive_bayes_combination, text_pair_feature_extractor_prepopulated, x_train, y_train, x_test, y_test, arg, pcc_test_param, raw_c_test)

    # with ProcessPoolExecutor() as executor:
    #     executor.submit(run_svm, clf_svm_flag, svm_combinations, text_pair_feature_extractor_prepopulated, x_train, y_train, x_test, y_test, arg, pcc_test_param, raw_c_test)
    #     executor.submit(run_lr, clf_lr_flag, lr_combinations, text_pair_feature_extractor_prepopulated, x_train, y_train, x_test, y_test, arg, pcc_test_param, raw_c_test)
    #     executor.submit(run_nb, clf_nb_flag, naive_bayes_combinations, text_pair_feature_extractor_prepopulated, x_train, y_train, x_test, y_test, arg, pcc_test_param, raw_c_test)

def run_experiment(arguments, _type):
    durations = []
    printLog.debug(f'Running {_type}-experiment with {len(arguments)} combination(s)')
    current_auroc = 0
    best_classifier = ''
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
        
        #current_auroc, best_classifier = get_best_auroc(arguments, current_auroc)

        printLog.info(f'Experiment {i+1} took {elapsed_time:.2f} seconds. Estimated time left: {eta} (H:M:S). - best auroc: {current_auroc} - best classifier: {best_classifier}')  

def pan20_b1_tests():
    from experiments.b1.pan20_baseline_1_LR import pan20_b1_LR_tfidf, pan20_b1_LR_dependency
    from experiments.b1.pan20_baseline_1_NB import pan20_b1_NB_tfidf, pan20_b1_NB_dependency
    from experiments.b1.pan20_baseline_1_SVM import pan20_b1_SVM_tfidf, pan20_b1_SVM_dependency
    
    _NAME = 'pan20_b1_LR_dependency'
    run_experiment(pan20_b1_LR_dependency(_NAME), _NAME)

    _NAME = 'pan20_b1_NB_dependency'
    run_experiment(pan20_b1_NB_dependency(_NAME), _NAME)

    _NAME = 'pan20_b1_SVM_dependency'
    run_experiment(pan20_b1_SVM_dependency(_NAME), _NAME)

    _NAME = 'pan20_b1_LR_tfidf'
    run_experiment(pan20_b1_LR_tfidf(_NAME), _NAME)

    _NAME = 'pan20_b1_NB_tfidf'
    run_experiment(pan20_b1_NB_tfidf(_NAME), _NAME) 

    _NAME = 'pan20_b1_SVM_tfidf'
    run_experiment(pan20_b1_SVM_tfidf(_NAME), _NAME)

def pan20_b15_test():
    from experiments.b15.pan20_baseline_1_LR import pan20_b15_LR_tfidf, pan20_b15_LR_dependency
    from experiments.b15.pan20_baseline_1_NB import pan20_b15_NB_tfidf, pan20_b15_NB_dependency
    from experiments.b15.pan20_baseline_1_SVM import pan20_b15_SVM_tfidf, pan20_b15_SVM_dependency

    _NAME = 'pan20_b15_LR_dependency'
    run_experiment(pan20_b15_LR_dependency(_NAME), _NAME)

    _NAME = 'pan20_b15_NB_dependency'
    run_experiment(pan20_b15_NB_dependency(_NAME), _NAME)

    _NAME = 'pan20_b15_SVM_dependency'
    run_experiment(pan20_b15_SVM_dependency(_NAME), _NAME)

    _NAME = 'pan20_b15_LR_tfidf'
    run_experiment(pan20_b15_LR_tfidf(_NAME), _NAME)

    _NAME = 'pan20_b15_NB_tfidf'
    run_experiment(pan20_b15_NB_tfidf(_NAME), _NAME) 

    _NAME = 'pan20_b15_SVM_tfidf'
    run_experiment(pan20_b15_SVM_tfidf(_NAME), _NAME)

def pan20_ra1_test():
    from experiments.ra1.pan20_ra1_LR import pan20_ra1_LR_tfidf, pan20_ra1_LR_dependency
    from experiments.ra1.pan20_ra1_SVM import pan20_ra1_SVM_tfidf, pan20_ra1_SVM_dependency

    _NAME = 'pan20_ra1_LR_lexical'
    run_experiment(pan20_ra1_LR_tfidf(_NAME), _NAME)

    _NAME = 'pan20_ra1_SVM_lexical'
    run_experiment(pan20_ra1_SVM_tfidf(_NAME), _NAME)

    _NAME = 'pan20_ra1_SVM_dependency'
    run_experiment(pan20_ra1_SVM_dependency(_NAME), _NAME)

    _NAME = 'pan20_ra1_LR_dependency'
    run_experiment(pan20_ra1_LR_dependency(_NAME), _NAME)

def main():
    #pan20_b1_tests()
    #pan20_b15_test()
    pan20_ra1_test()

if __name__ == '__main__':
    main()