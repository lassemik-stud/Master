from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import classification_report
import pandas as pd
import pickle

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

x_train, y_train, x_test, y_test, PCC_train_params, PCC_test_params = load_or_process_data(cutoff=100,sentence_size=50,k=5,d=2)

# Initialize the pipeline
PCC_Pipeline = Pipeline([
                ('feature_extractor', TextPairFeatureExtractor(ngram_range=(4,4),max_features=1000,feature_type='tfidf',analyzer='char')),
                ('scaler', StandardScaler()),
                ('classifier', SVC(C=1, kernel='sigmoid', gamma='scale', degree=1,probability=True))
            ])


# Preload the data if it has been processed before
root_path = '../pre_data/'
if data_exists(root_path+'y_pred.pkl'):
    y_pred = load_data_from_pickle(root_path+'y_pred.pkl')
    printLog.debug('Classification data loaded from pickle files')
else:
    PCC_Pipeline.fit(x_train, y_train)
    y_pred = PCC_Pipeline.predict_proba(x_test)
    save_data_to_pickle(y_pred, root_path+'y_pred.pkl')
    printLog.debug('Classification data saved to pickle files')

map_test_pairs_to_c = []

pcc_i = 0
print(y_test)
for pair_i, element in enumerate(PCC_test_params):
    c_size = int(element['l'])
    y_out = ([sublist[1] for sublist in y_pred[pcc_i:pcc_i+c_size]])
    y_truth = y_test[pcc_i:pcc_i+c_size]
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

    print(f'RESULTS FOR {pair_i}')
    for i, part in enumerate(N_val[:N]):
        array_sum = sum(part)
        count_zero = part.count(0)
        count_ones = part.count(1)
        result = array_sum if array_sum == 0 else array_sum / len(part)
        print(f'--> {round(result,2)} - {y_truth[i]} -\t Weight: {(count_ones-count_zero)/l_group}')
    print("-----------------------")



    pcc_i+=c_size
