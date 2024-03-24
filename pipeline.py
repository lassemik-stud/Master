from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import spacy
from sklearn.metrics import classification_report
import random
import json 
import datetime
import pandas as pd

from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm

from tqdm import tqdm
from settings.logging import printLog
from multiprocessing import Pool, cpu_count
from preprocess import load_or_process_data

nlp = spacy.load('en_core_web_sm')

# Custom transformer to process pairs of texts
class TextPairFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self,ngram_range,max_features,feature_type,analyzer):
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.feature_type = feature_type
        self.analyzer = analyzer
        
        # Initialize with whatever feature extractors you plan to use
        if self.feature_type == 'tfidf':
            self.vectorizer = TfidfVectorizer(tokenizer=None,ngram_range=self.ngram_range,max_features=self.max_features,analyzer=self.analyzer)
        elif self.feature_type == 'BoW':
            self.vectorizer = CountVectorizer(tokenizer=None,ngram_range=self.ngram_range,max_features=self.max_features,analyzer=self.analyzer)
    
    def fit(self, X, y=None):
        #printLog.debug(f'Feature extractor initialized with ngram_range={self.ngram_range}, max_features={self.max_features}, feature_type={self.feature_type}, analyzer={self.analyzer}')
        # Fit the vectorizer on all texts, assuming X is a list of text pairs (tuples)
        all_texts = [text for pair in X for text in pair]
        self.vectorizer.fit(all_texts)
        return self
    
    def transform(self, X, y=None):
        # Transform each pair of texts to a feature difference or similarity measure
        features = []
        for text1, text2 in X:
            feat1 = self.vectorizer.transform([text1]).toarray()
            feat2 = self.vectorizer.transform([text2]).toarray()
            # Example: simple difference of feature vectors
            feat_diff = np.abs(feat1 - feat2)
            features.append(feat_diff.flatten())
        return np.array(features)



def run_pipeline(feature_extractor_ngram_range, feature_extractor_max_features, feature_type,feature_analyzer,sentence_size,samples,no_load_flag): 
    pipeline = {
        'pipeline_SVM': {
            'pipeline': Pipeline([
                ('feature_extractor', TextPairFeatureExtractor(ngram_range=feature_extractor_ngram_range,max_features=feature_extractor_max_features,feature_type=feature_type,analyzer=feature_analyzer)),
                ('scaler', StandardScaler()),
                ('classifier', SVC())
            ]),
            'parameters': {
                'classifier__C': [0.1,1,10],
                'classifier__kernel': ['sigmoid'],
                'classifier__gamma': ['scale'],
                'classifier__degree': [1]
            }
        },
        'pipeline_LR' : {
        'pipeline' : Pipeline([
                            ('feature_extractor', TextPairFeatureExtractor(ngram_range=feature_extractor_ngram_range,max_features=feature_extractor_max_features,feature_type=feature_type,analyzer=feature_analyzer)),
                            ('scaler', StandardScaler()),
                            ('classifier', LogisticRegression())]),
        'parameters' : {
                        'classifier__C': [0.01, 0.1, 1, 10],
                        'classifier__penalty': ['elasticnet'],
                        'classifier__solver': ['saga'],
                        'classifier__l1_ratio': np.linspace(0, 1, 10),
                        'classifier__max_iter': [2000]
                        }
                    }
    }
    
    x_train, y_train, x_test, y_test = load_or_process_data(samples,sentence_size=sentence_size,no_load_flag=True)
    printLog.debug(f'after load: sizes: x_train: {len(x_train)}, y_train: {len(y_train)}, x_test: {len(x_test)}, y_test: {len(y_test)}')
    
    scores = []

    for _pipeline, mp in pipeline.items():
        printLog.debug(f'Grid search started - {_pipeline}')
        grid_pipeline = GridSearchCV(mp['pipeline'], mp['parameters'],n_jobs=-1,verbose=1,)
        grid_pipeline.fit(x_train, y_train)
        scores.append([
            _pipeline,
            grid_pipeline.best_score_,
            grid_pipeline.best_params_,
            feature_extractor_ngram_range,
            feature_extractor_max_features,
            feature_type,
            feature_analyzer,
            samples,
            sentence_size]
        )
        printLog.debug(f'Grid search completed - {_pipeline}')

    #now = datetime.datetime.now()
    #timestamp = now.strftime('%Y-%m-%d-%H-%M-%S')
    #filename = f'output/{timestamp}-grid_search_results.csv'

    #df = pd.DataFrame(scores,columns=['model','best_score','best_params','ngram_range','max_features','feature_type','analyzer'])
    #df.to_csv(filename, index=False)
    return scores
    
# Grid parameters and defining pipelines
def grid_search():
    scores_tot = []
    i = 0
    feature_extractor_ngram_range = [(1,1),(2,2),(3,3),(4,4)]
    feature_extractor_max_features = [1000]
    feature_type = ['tfidf','BoW'] 
    feature_analyzer = ['char']
    sample_size = [50,100,1000]
    sentence_size = [20,50,300,500]
    old_sentence_size = sentence_size[0]
    tot_iterations = len(feature_extractor_ngram_range)*len(feature_extractor_max_features)*len(feature_type)*len(feature_analyzer)*len(sample_size)*len(sentence_size)
    printLog.debug(f'Starting. Total iterations - {tot_iterations}')
    for s_size in sample_size:
        for sen_size in sentence_size:
            
            for ngram_range in feature_extractor_ngram_range:
                for max_features in feature_extractor_max_features:
                    for feature in feature_type:
                        for analyzer in feature_analyzer:
                            if sen_size != old_sentence_size:
                                no_load_flag = True
                            else: 
                                no_load_flag = False
                            i+=1
                            time_start = datetime.datetime.now()
                            scores_tot.extend(run_pipeline(ngram_range, max_features, feature, analyzer,sen_size,s_size,no_load_flag=no_load_flag))
                            time_end = datetime.datetime.now()
                            total_time = time_end - time_start
                            
                            printLog.debug(f'Ran pipeline {i} of {tot_iterations} - ETA \t {datetime.datetime.now() + (tot_iterations-i)*total_time}')
                            old_sentence_size = sen_size

    df_tot = pd.DataFrame(scores_tot, columns=['model', 'best_score', 'best_params', 'ngram_range', 'max_features', 'feature_type', 'analyzer', 'samples','sentence_size'])
    now = datetime.datetime.now()
    timestamp = now.strftime('%Y-%m-%d-%H-%M-%S')
    filename = f'output/{timestamp}-tot-grid_search_results.csv'
    pd.set_option('display.max_columns', None)  # to display all columns
    pd.set_option('display.expand_frame_repr', False)  # to display all rows
    pd.set_option('display.max_colwidth', None)  # to display full content of each cell
    df_tot.to_csv(filename, index=False,)
    print(df_tot)

def SVM_pipe(feature_extractor_ngram_range,feature_type, feature_analyzer,feature_extractor_max_features):
    return Pipeline([
                ('feature_extractor', TextPairFeatureExtractor(ngram_range=feature_extractor_ngram_range,max_features=feature_extractor_max_features,feature_type=feature_type,analyzer=feature_analyzer)),
                ('scaler', StandardScaler()),
                ('classifier', SVC(C=1, kernel='sigmoid', gamma='scale', degree=1))
            ])

def LR_pipe(feature_extractor_ngram_range,feature_type, feature_analyzer,feature_extractor_max_features):
    return Pipeline([
                ('feature_extractor', TextPairFeatureExtractor(ngram_range=feature_extractor_ngram_range,max_features=feature_extractor_max_features,feature_type=feature_type,analyzer=feature_analyzer)),
                ('scaler', StandardScaler()),
                ('classifier', LogisticRegression(C=1, penalty='elasticnet', solver='saga', l1_ratio=0.333333333, max_iter=1000))
            ])

def run_pipeline_single(feature_extractor_ngram_range, feature_extractor_max_features, feature_type,feature_analyzer):
    samples = 1000
    x_train, y_train, x_test, y_test = load_or_process_data(samples,300)
    printLog.debug(f'after load: sizes: x_train: {len(x_train)}, y_train: {len(y_train)}, x_test: {len(x_test)}, y_test: {len(y_test)}')

    svm_pipe = SVM_pipe(feature_extractor_ngram_range,feature_type, feature_analyzer,feature_extractor_max_features)
    printLog.debug(f'Running SVM pipeline - fit')
    svm_pipe.fit(x_train, y_train)
    printLog.debug(f'Running SVM pipeline - predict')
    svm_y_pred = svm_pipe.predict(x_test)
    print(classification_report(y_test, svm_y_pred))

def main():
    #run_pipeline_single(feature_extractor_ngram_range=(3,3),feature_type='tfidf',feature_analyzer='char',feature_extractor_max_features=1000)
    grid_search()

if __name__ == '__main__':
    main()