from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import classification_report
import pandas as pd
import pickle
from concurrent.futures import ProcessPoolExecutor

from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm

import tqdm
from settings.logging import printLog
from multiprocessing import Pool, cpu_count

from feature_extraction import TextPairFeatureExtractor
from preprocess import load_or_process_data, save_data_to_pickle, load_data_from_pickle, data_exists
threashold = 0.5
c_value = 0.01
on_change_value = 0.5
cutoff=100
sentence_size = 30
k=5
d=2


def round_by_threashold(value,c):
    global threashold
    if c - on_change_value > value:
        threashold+=c_value
    elif c < value - on_change_value:
        threashold-=c_value
    return 1 if value > threashold else 0
def run_pipeline(args):
    feature_extractor_ngram_range, feature_extractor_max_features, feature_type,feature_analyzer,sentence_size,samples,k,d,svc_c,svc_degree = args
    cutoff=samples
    if k < d: 
        print(f'k={k} is less than d={d}, skipping')
        return 0

    x_train, y_train, x_test, y_test, PCC_train_params, PCC_test_params = load_or_process_data(cutoff=cutoff,sentence_size=sentence_size,k=k,d=d)

    # Initialize the pipeline
    PCC_Pipeline = Pipeline([
                    ('feature_extractor', TextPairFeatureExtractor(ngram_range=feature_extractor_ngram_range,max_features=feature_extractor_max_features,feature_type='tfidf',analyzer='char')),
                    ('scaler', StandardScaler()),
                    ('classifier', SVC(C=svc_c, kernel='sigmoid', gamma='scale', degree=svc_degree, probability=True))
                ])

    # Preload the data if it has been processed before
    root_path = '../pre_data/'
    path = f'{root_path}-svc_c-{svc_c}-{svc_degree}-{str(cutoff)}-{str(sentence_size)}-{k}-{d}-ft-{feature_type}-fa{feature_analyzer}-ngram-{str(feature_extractor_ngram_range[0])}-{str(feature_extractor_ngram_range[1])}-{feature_extractor_max_features}-y_pred.pkl'
    if data_exists(path):
        y_pred = load_data_from_pickle(path)
        printLog.debug('Classification data loaded from pickle files')
    else:
        PCC_Pipeline.fit(x_train, y_train)
        y_pred = PCC_Pipeline.predict_proba(x_test)
        save_data_to_pickle(y_pred, path)
        printLog.debug('Classification data saved to pickle files')

    map_test_pairs_to_c = []

    pcc_i = 0
    y_test_pred = []
    y_test_transformed = []

    for pair_i, element in enumerate(PCC_test_params):
        
        c_size = int(element['l'])
        y_out = ([sublist[1] for sublist in y_pred[pcc_i:pcc_i+c_size]])
        
        y_truth = y_test[pcc_i:pcc_i+c_size]
        #print(y_truth)
        

        n = element['n']
        k = element['k']
        d = element['d']
        l = element['l']
        N = element['N']
        
        l_group = element['l_group']

        N_theoretical = (n-1)*(k-d) + k
        
        N_val = [[] for _ in range(N_theoretical)]

        count_c = 0
        for elements in range(int(l/n)):
            j = 0
            for element in range(n):
                for i in range(k):
                    N_val[j].append(y_out[count_c])
                    j+=1
                
                count_c+=1
                j-=d
        
        #print(f'RESULTS FOR {pair_i}')
        sum_c = []
        length = 0
        for i, part in enumerate(N_val[:N]):
            array_sum = sum(part)
            length = len(part)
            result = array_sum if array_sum == 0 else array_sum / length
            sum_c.append(result)
            #print(f'--> {round(result,2)} - {y_truth[i]}')
        #print("-----------------------")
        #print(len(y_truth), len(y_test),pcc_i, c_size, N)
        y_test_pred.append(round_by_threashold(sum(sum_c)/length,y_truth[0]))
        y_test_transformed.append(y_truth[0])
        
        #print("-----------------------")



        pcc_i+=c_size
    res = classification_report(y_test_transformed, y_test_pred)
    print(res)
   
    with open(f'/home/lasse/Master/results/report-s{samples}-se{sentence_size}-fa{feature_analyzer}-ft-{feature_type}-fng{feature_extractor_ngram_range[0]}{feature_extractor_ngram_range[1]}-fm{feature_extractor_max_features}-k{k}-d{d}-svm-c{svc_c}-d{svc_degree}.txt', 'w') as f:
        f.write(res)

feature_extractor_ngram_range = [(4,4)]
feature_extractor_max_features = [1000]
feature_type = ['tfidf']
feature_analyzer = ['char']
sentence_size = [30, 40, 50, 100]
samples = [100]
k = [4, 5]
d = [1, 2, 3, 4]
svc_c = [1,10,100]
svc_degree = [1]

args = [
    (first, second, third, fourth, fifth, sixth, seventh, eighth)
    for first in feature_extractor_ngram_range
    for second in feature_extractor_max_features
    for third in feature_type
    for fourth in feature_analyzer
    for fifth in sentence_size
    for sixth in samples
    for seventh in k
    for eighth in d
]

printLog.debug(f'Running {len(args)} iterations')
# Create a pool of workers
with ProcessPoolExecutor() as executor:
    executor.map(run_pipeline, args)