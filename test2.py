
import os

from steps.A1_preprocessing.A1_parse import Corpus, Instance

from settings.static_values import EXPECTED_PREPROCESSED_DATASETS_FOLDER as EXP_PRE_DATASETS_FOLDER
from steps.setup import setup

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize

from sklearn.model_selection import GridSearchCV

from scipy.sparse import hstack

# Define a range of parameter values to test
param_grid = {
    'oneclasssvm__nu': [0.1, 0.5, 0.7],
    'oneclasssvm__gamma': ['scale', 'auto', 0.01, 0.1, 0.5],
    'oneclasssvm__kernel': ['rbf', 'poly', 'sigmoid']
}
def test_setup():
    setup()

def test_A1_A1_parsing():
    corpus = Corpus()
    corpus.parse_raw_data(os.path.join(EXP_PRE_DATASETS_FOLDER,"pan13-train.jsonl"))
    #corpus.split_corpus(0.7,0.15)
    #corpus.get_avg_statistics()
    #corpus.print_corpus_info()
    return corpus

class fullExperiment():
    def __init__(self, corpus:Corpus, model):
        self.corpus = corpus
        self.predictions = []
        self.true_labels = []
        self.model = model
    
    def evaluations(self):
        accuracy = accuracy_score(self.true_labels, self.predictions)
        precision = precision_score(self.true_labels, self.predictions)
        recall = recall_score(self.true_labels, self.predictions)
        f1 = f1_score(self.true_labels, self.predictions)
        return accuracy
        #print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}")

    def run(self):
        for instance in self.corpus.all:
            if 'EN' not in instance.author:
                continue
            se = singleExperiment(instance, self.model)
            se.extract_features_both()
            #se.manual_grid_search()
            se.train_model_svm()
            se.predict()
            self.predictions.append(se.prediction[0])
            self.true_labels.append(se.true_label)
            #print(f"{instance.author} - Prediction: {se.prediction[0]}, True Label: {se.true_label}")
        #break
            
        return self.evaluations()

class singleExperiment():
    def __init__(self, instance:Instance, model):
        self.instance = instance

        # Parameters for manual grid search
       

        # Features
        self.kt_train_vectors = []
        self.ut_train_vectors = []
        self.vectorizer = TfidfVectorizer(ngram_range=(1,2))

        # Train 
        self.model = model

        # Prediction
        self.prediction = 0
        self.true_label = self.instance.same_author
    
    def extract_features_bow(self):
        kt_texts = [" ".join(self.instance.known_text)]
        #kt_texts = [self.instance.known_text] if isinstance(self.instance.known_text, str) else self.instance.known_text
        ut_texts = [self.instance.unknown_text] if isinstance(self.instance.unknown_text, str) else self.instance.unknown_text
        self.kt_train_vectors = self.vectorizer.fit_transform(kt_texts)
        self.ut_train_vectors = self.vectorizer.transform(ut_texts)

    def extract_features_both(self):
        self.extract_features_bow()
        kt_features = self.extract_stylistic_features(" ".join(self.instance.known_text))
        ut_features = self.extract_stylistic_features(" ".join(self.instance.unknown_text))
        self.kt_train_vectors = self.combine_features(self.kt_train_vectors, kt_features)
        self.ut_train_vectors = self.combine_features(self.ut_train_vectors, ut_features)

    def train_model_svm(self):
        self.model.fit(self.kt_train_vectors)

    def predict(self):
        self.prediction = self.model.predict(self.ut_train_vectors)
        

    @staticmethod
    def extract_stylistic_features(text):
        words = word_tokenize(text)
        sentences = sent_tokenize(text)
        
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        avg_sentence_length = np.mean([len(word_tokenize(sentence)) for sentence in sentences]) if sentences else 0
        type_token_ratio = len(set(words)) / len(words) if words else 0
        
        return np.array([avg_word_length, avg_sentence_length, type_token_ratio])
    
    @staticmethod
    def combine_features(tfidf_vector, stylistic_features):
        # Assuming tfidf_vector is a sparse matrix and stylistic_features is a numpy array
        return hstack([tfidf_vector, stylistic_features.reshape(1, -1)])

corpus = test_A1_A1_parsing()

def manual_grid_search():
    nu_values = [0.1, 0.5, 0.7]
    gamma_values = ['scale', 'auto', 0.01, 0.1, 0.5]
    kernel_values = ['rbf', 'poly', 'sigmoid']
    best_score = -np.inf
    best_params = {}
    for nu in nu_values:
        for gamma in gamma_values:
            for kernel in kernel_values:
                # Re-initialize model with current parameters
                model = make_pipeline(StandardScaler(with_mean=False), svm.OneClassSVM(nu=nu, kernel=kernel, gamma=gamma))
                # Train and predict using the current model setup
                
                fullexp = fullExperiment(corpus, model)
                current_score = fullexp.run()
                # Implement your evaluation logic here to calculate the 'current_score'
                # For demonstration, let's assume 'current_score' is obtained
                # NOTE: You'll need to adapt this part to fit your evaluation strategy
                
                # Example: current_score = your_evaluation_method(...)
                # For now, we'll simulate 'current_score' with a placeholder
                # Placeholder for actual score calculation
                
                # Update best model if current model is better
                if current_score > best_score:
                    best_score = current_score
                    best_params = {'nu': nu, 'gamma': gamma, 'kernel': kernel}
    print(f"Best Parameters: {best_params}, Best Score: {best_score}")

manual_grid_search()

