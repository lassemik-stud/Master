
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

# NEW PROJECT
from nltk import tokenize
from nltk import tag
from nltk import RegexpParser
from nltk import tree
from nltk import data, download
import nltk
from nltk import pos_tag
from nltk.corpus import stopwords
import readability
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
import re
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from settings.logging import printLog as print_log

# Feature importance 
import matplotlib.pyplot as plt
import seaborn as sns

# Feature extraction of dependency parse tree
import spacy
from sklearn.base import BaseEstimator, TransformerMixin
nlp = spacy.load('en_core_web_sm')

# word2vec - load the model. 
# https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing
# https://github.com/harmanpreet93/load-word2vec-google
#from gensim.models import KeyedVectors
#word2vec_model_path = '..\pretrained_models\GoogleNews-vectors-negative300.bin\GoogleNews-vectors-negative300.bin'
#word2vec_model = KeyedVectors.load_word2vec_format(word2vec_model_path, binary=True)

# word2vec imports 
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# nltk.download('treebank')

# permutation importance
from sklearn.inspection import permutation_importance

chunker_instance = None
tokenizer = tokenize.TreebankWordTokenizer()
#tagger = data.load("treebank")

# Import all stuffs for NLTK
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

import multiprocessing as mp
from multiprocessing import cpu_count

grammar = r"""
  NP: 
      {<DT|WDT|PP\$|PRP\$>?<\#|CD>*(<JJ|JJS|JJR><VBG|VBN>?)*(<NN|NNS|NNP|NNPS>(<''><POS>)?)+}
      {<DT|WDT|PP\$|PRP\$><JJ|JJS|JJR>*<CD>}
      {<DT|WDT|PP\$|PRP\$>?<CD>?(<JJ|JJS|JJR><VBG>?)}
      {<DT>?<PRP|PRP\$>}
      {<WP|WP\$>}
      {<DT|WDT>}
      {<JJR>}
      {<EX>}
      {<CD>+}
  VP: {<VBZ><VBG>}
      {(<MD|TO|RB.*|VB|VBD|VBN|VBP|VBZ>)+}
      

"""

class Instance():
    def __init__(self):
        self.id = 0
        self.dataset = 0
        self.type = ""
        self.author = ""
        self.same_author = 0
        self.known_text = set()
        self.unknown_text = set()
        self.additional_info = ""

        # Features
        self.token_kt = []
        self.token_pt = []
        self.bow_kt = {}
        self.bow_pw = {}       
    
    def __str__(self):
        return f"ID: {self.id}, Dataset: {self.dataset}, Type: {self.type}, Author: {self.author}, Same Author: {self.same_author}, Known Text: {self.known_text}, Additional Info: {self.additional_info}"


def load_corpus(dataset, max_entries = 0):
    corpus = Corpus()
    corpus.parse_raw_data(os.path.join(EXP_PRE_DATASETS_FOLDER,dataset))
    corpus.select_balanced_corpus(max_entries)
    corpus.get_avg_statistics()
    corpus.print_corpus_info()
    return corpus

class fe():
    def __init__(self, raw_corpus_train:Corpus, raw_corpus_test:Corpus):
        self.raw_corpus = {'train': raw_corpus_train,'test': raw_corpus_test}
        self.x_corpus = {'train': [],'test': []}
        self.y_truth = {'train': [],'test': []}
        self.x_features_diff = {'train': [],'test': []}
        self.test_predicions = []

        # transformer and scalers
        self.transformer = get_writeprints_transformer()
        self.scaler = StandardScaler(with_mean=False)

        # Features
        self.x_kn = {'train': [],'test': []}
        self.x_un = {'train': [],'test': []}

        # Classifier
        self.clf = LogisticRegression()

    def run(self):
        print_log.debug("Experiment 1 started...")
        #self.run_feature_extraction_backup()
        self.run_feature_extraction()
        self.fit_to_transformer()
        self.vectorize_corpus()
        #self.assess_permutation_importance()
        test_pred = self.classify_LR()
        self.evaluations()
        #self.assess_feature_importance()
        #self.assess_feature_importance_aggregate()
        
    def evaluations(self):
        print_log.debug("Evaluating:")        
        accuracy = accuracy_score(self.y_truth['test'], self.test_predicions)
        precision = precision_score(self.y_truth['test'], self.test_predicions)
        recall = recall_score(self.y_truth['test'], self.test_predicions)
        f1 = f1_score(self.y_truth['test'], self.test_predicions)
        print_log.debug(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}")
    
    def classify_LR(self):
        threshold = 0.6
        print_log.debug("Classifying with Logistic Regression...")
        self.clf = LogisticRegression(random_state=42).fit(self.x_features_diff['train'], self.y_truth['train'])
        test_predicions = self.clf.predict_proba(self.x_features_diff['test'])[:, 1]
        self.test_predicions = [1 if prob > threshold else -1 for prob in test_predicions]
        
    def print_test(self):
        print(self.x_corpus['train'][0]['id'])
        print(self.y_truth['train'][0])

    def fit_to_transformer(self):
        print_log.debug("Fitting to transformer...")
        fit_corpus = [item['kt'] for item in self.x_corpus['train']] + [item['ut'] for item in self.x_corpus['train']]
        corpus_transformed = self.transformer.fit_transform(fit_corpus[:len(fit_corpus)//2])
        corpus_scaled = self.scaler.fit_transform(corpus_transformed) 

    def vectorize_corpus(self):
        for version in 'train', 'test':
            print_log.debug(f"Vectorizing {version}...")
            self.x_kn[version] = self.scaler.transform(self.transformer.transform([item['kt'] for item in self.x_corpus[version]]))
            self.x_un[version] = self.scaler.transform(self.transformer.transform([item['ut'] for item in self.x_corpus[version]]))
            self.x_features_diff[version] = pd.DataFrame(np.abs(self.x_kn[version] - self.x_un[version]).todense())
            print_log.debug(f"Length of vectorized {version}: {len(self.x_features_diff[version])}")

    def run_feature_extraction_backup(self):
        for version in 'train', 'test':
            print_log.debug(f"Extracting features for {version}...")
            for instance in self.raw_corpus[version].all:
                id = instance.id
                #kt = fe.feature_extraction(" ".join(instance.known_text))
                kt = fe.feature_extraction(instance.known_text[0])
                ut = fe.feature_extraction(instance.unknown_text[0])
                truth = 1 if int(instance.same_author) == 1 else -1
                self.x_corpus[version].append({'id': id, 'kt': kt, 'ut': ut})
                self.y_truth[version].append(truth)

    @staticmethod
    def worker(instance):
        id = instance.id
        kt = fe.feature_extraction(instance.known_text[0])
        #kt = fe.feature_extraction(" ".join(instance.known_text)) # used for PAN13 that has multiple known texts
        ut = fe.feature_extraction(instance.unknown_text[0])
        truth = 1 if instance.same_author == 1 else -1
        return {'id': id, 'kt': kt, 'ut': ut}, truth

    def run_feature_extraction(self):
        for version in 'train', 'test':
            print_log.debug(f"Extracting features for {version}...")
            with mp.Pool(cpu_count() - 1) as pool:
                results = pool.map(self.worker, self.raw_corpus[version].all)
            for result in results:
                self.x_corpus[version].append(result[0])
                self.y_truth[version].append(result[1])  
              
            
    
    @staticmethod
    def feature_extraction(text): 
        tokens = []
        for token in tokenizer.tokenize(text):
            tokens.append(token)
        
        tagger_output = pos_tag(tokens)
        pos_tags = [token[1] for token in tagger_output]
        pos_chunks,subtree_expansions = nltk_pos_tag_chunk(tagger_output)

        entry = {
            'preprocessed': text,
            'pos_tags': pos_tags,
            'pos_tag_chunks': pos_chunks,
            'pos_tag_chunk_subtrees': subtree_expansions,
            'tokens': tokens
        }
        return entry

    def assess_feature_importance(self):
        print_log.debug("Assessing feature importance...")
        coefficients = self.clf.coef_[0]
        feature_names = get_features_from_FeatureUnion(transformer_union=self.transformer)
        importances = sorted(zip(feature_names, coefficients), key=lambda x: abs(x[1]), reverse=True)
        importances_df = pd.DataFrame(importances, columns=['Feature', 'Coefficient'])
        plt.figure(figsize=(10,8))
        sns.barplot(data=importances_df.head(50),x='Coefficient',y='Feature')
        plt.title('Top 20 Feature Importance from LR')
        plt.savefig("plot/fi_1.png")
        
    def assess_feature_importance_aggregate(self):
        print_log.debug("Assessing feature importance...")
        coefficients = self.clf.coef_[0]
        feature_names = get_features_from_FeatureUnion(transformer_union=self.transformer)

        # Aggregate coefficients by transformer prefix
        transformer_impact = {}
        for name, coef in zip(feature_names, coefficients):
            # Extract the transformer prefix (e.g., 'char_dist')
            prefix = name.split('__')[0]
            # Sum the absolute values of coefficients for each prefix
            if prefix in transformer_impact:
                transformer_impact[prefix] += abs(coef)
            else:
                transformer_impact[prefix] = abs(coef)

        # Convert the dictionary to a list of tuples and sort it
        importances = sorted(transformer_impact.items(), key=lambda x: x[1], reverse=True)
        importances_df = pd.DataFrame(importances, columns=['Transformer', 'Total Impact'])

        # Plot the results
        plt.figure(figsize=(10, 8))
        sns.barplot(data=importances_df.head(20), x='Total Impact', y='Transformer')
        plt.title('Top Transformer Feature Importance from LR')
        plt.savefig("plot/fi_2.png")

    def assess_permutation_importance(self):
        # Assuming you've already split your dataset into X_train, X_test, y_train, y_test
        # and that you've transformed X_test using the same transformations as X_train
        transformed_X_test = self.transformer.transform(self.x_kn['train'])
        feature_names = get_features_from_FeatureUnion(self.transformer)
        
        # Compute permutation importance
        r = permutation_importance(self.clf, transformed_X_test, self.y_truth['train'], n_repeats=30, random_state=42, n_jobs=-1)
        
        # Sort the features by importance
        sorted_idx = r.importances_mean.argsort()

        # Plot
        plt.figure(figsize=(12, 8))
        plt.boxplot(r.importances[sorted_idx].T, vert=False, labels=np.array(feature_names)[sorted_idx])
        plt.title("Permutation Importance (test set)")
        plt.tight_layout()
        #plt.show()

def get_nltk_pos_tag_based_chunker():
    global chunker_instance
    if chunker_instance is not None:
        return chunker_instance
    chunker_instance = RegexpParser(grammar)
    return chunker_instance


def nltk_pos_tag_chunk(pos_tags):
    chunker = get_nltk_pos_tag_based_chunker()
    parse_tree = chunker.parse(pos_tags)
    subtree_expansions = []
    for subt in parse_tree:
        extract_subtree_expansions(subt, subtree_expansions)
    return list(map(chunk_to_str, parse_tree)), subtree_expansions


def extract_subtree_expansions(t, res):
    if type(t) is tree.Tree:
        expansion = t.label() + "[" + " ".join([chunk_to_str(child) for child in t]) + "]"
        res.append(expansion)
        for child in t:
            extract_subtree_expansions(child, res)


def chunk_to_str(chunk):
    if type(chunk) is tree.Tree:
        return chunk.label()
    else:
        return chunk[1]

class CustomTfIdfTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, key, analyzer, n=1, vocab=None):
        self.key = key
        if self.key == 'pos_tags' or self.key == 'tokens' or self.key == 'pos_tag_chunks' or self.key == 'pos_tag_chunk_subtrees':
            self.vectorizer = TfidfVectorizer(analyzer=analyzer, min_df=0.1, tokenizer=pass_fn, preprocessor=pass_fn, vocabulary=vocab, norm='l1', ngram_range=(1, n), token_pattern=None)
        else:
            self.vectorizer = TfidfVectorizer(analyzer=analyzer, min_df=0.1, vocabulary=vocab, norm='l1', ngram_range=(1, n))

    def fit(self, x, y=None):
        self.vectorizer.fit([entry[self.key] for entry in x], y)
        return self

    def transform(self, x):
        return self.vectorizer.transform([entry[self.key] for entry in x])
    
    def get_feature_names_out(self):
        return self.vectorizer.get_feature_names_out()
    
    
class CustomFreqTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, analyzer, n=1, vocab=None):
        self.vectorizer = TfidfVectorizer(tokenizer=pass_fn, preprocessor=pass_fn, vocabulary=vocab, norm=None, ngram_range=(1, n), token_pattern=None)

    def fit(self, x, y=None):
        self.vectorizer.fit([entry['tokens'] for entry in x], y)
        return self

    def transform(self, x):
        d = np.array([1 + len(entry['tokens']) for entry in x])[:, None]
        return self.vectorizer.transform([entry['tokens'] for entry in x]) / d
    
    def get_feature_names_out(self):
        return self.vectorizer.get_feature_names_out()
    
    
class CustomFuncTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, transformer_func, fnames=None):
        self.transformer_func = transformer_func
        self.fnames = fnames
        
    def fit(self, x, y=None):
        return self
    
    def transform(self, x):
        xx = np.array([self.transformer_func(entry) for entry in x])
        if len(xx.shape) == 1:
            return xx[:, None]
        else:
            return xx
    
    def get_feature_names_out(self):
        if self.fnames is None:
            return ['']
        else:
            return self.fnames
        
def get_features_from_FeatureUnion(transformer_union):
    feature_names = []
    for transformer_name, transformer in transformer_union.transformer_list:
        try:
            names = [f"{transformer_name}__{name}" for name in transformer.get_feature_names_out()]
            feature_names.extend(names)
        except Exception as e:
            print_log.error(f"Error getting feature names from: {transformer_name} - {e}")
    return feature_names

def get_writeprints_transformer():
    char_distr = CustomTfIdfTransformer('preprocessed', 'char_wb', n=6)
    word_distr = CustomTfIdfTransformer('preprocessed', 'word', n=3)
    pos_tag_distr = CustomTfIdfTransformer('pos_tags', 'word', n=3)
    pos_tag_chunks_distr = CustomTfIdfTransformer('pos_tag_chunks', 'word', n=3)
    pos_tag_chunks_subtree_distr = CustomTfIdfTransformer('pos_tag_chunk_subtrees', 'word', n=1)
    #word_embeddings = Word2VecVectorizer(word2vec_model=word2vec_model)
    #char_distr = [('char_distr_{0}'.format(i), CustomTfIdfTransformer('preprocessed', 'char_wb', n=i)) for i in range(1,10)]
    #word_distr =  [('word_distr_{0}'.format(i), CustomTfIdfTransformer('preprocessed', 'word', n=i)) for i in range(1,10)]
    #pos_tag_distr =  [('pos_tags_{0}'.format(i), CustomTfIdfTransformer('pos_tags', 'word', n=i)) for i in range(1,10)]
    #pos_tag_chunks_distr =  [('pos_tag_chunks_distr_{0}'.format(i), CustomTfIdfTransformer('pos_tag_chunks', 'word', n=i)) for i in range(1,10)]
    #pos_tag_chunks_subtree_distr =  [('pos_tag_chunks_subtree_distr_{0}'.format(i), CustomTfIdfTransformer('pos_tag_chunk_subtrees', 'word', n=i)) for i in range(1,10)]
    
    # COPILOT FOUND THIS IS PUBLIC CODE --> https://github.com/github-copilot/code_referencing?cursor=ca5f9bd57d3db90e19fffc1d295dd071&editor=vscode
    # This is from PAN20 https://github.com/GabrielePisciotta/NLP-Authorship-Verification-Case-Study/blob/main/HumanLanguageTechnologies_project_2021_extended.ipynb
    # https://github.com/GabrielePisciotta/NLP-Authorship-Verification-Case-Study/blob/main/writeprints-extended.py#L61
    punctuation = '!"#$%&\'()*+,-./:;<=>?@[\]^_`{¦}~'
    special_char_distr = CustomTfIdfTransformer('preprocessed', 'char_wb', vocab=punctuation)
    freq_func_words = CustomFreqTransformer('word', vocab=stopwords.words('english'))

    transformer = FeatureUnion(
        #char_distr + 
        #word_distr +
        #pos_tag_distr +
        #pos_tag_chunks_distr +
        #pos_tag_chunks_subtree_distr +
        [
        ('char_distr', char_distr),
        ('word_distr', word_distr),
        ('pos_tag_distr', pos_tag_distr),
        ('pos_tag_chunks_distr', pos_tag_chunks_distr),
        ('pos_tag_chunks_subtree_distr', pos_tag_chunks_subtree_distr),
        #('word_embeddings', word_embeddings),
        ('special_char_distr', special_char_distr),
        ('freq_func_words', freq_func_words),
        ('hapax_legomena', CustomFuncTransformer(hapax_legomena)),
        ('character_count', CustomFuncTransformer(character_count)),
        ('distr_chars_per_word', CustomFuncTransformer(distr_chars_per_word, fnames=[str(i) for i in range(10)])),
        ('avg_chars_per_word', CustomFuncTransformer(avg_chars_per_word)),
        ('word_count', CustomFuncTransformer(word_count)),
        #('readability', CustomFuncTransformer(gunning_fog_index) ),
        ('n_of_pronouns', CustomFuncTransformer(n_of_pronouns)),
        #('dependency_features', DependencyFeatures())
    ], n_jobs=-1)
    
    return transformer

def word_count(entry):
    return len(entry['tokens'])

def avg_chars_per_word(entry):
    r = np.mean([len(t) for t in entry['tokens']])
    return r

def distr_chars_per_word(entry, max_chars=10):
    counts = [0] * max_chars
    for t in entry['tokens']:
        l = len(t)
        if l <= max_chars:
            counts[l - 1] += 1
    r = [c/len(entry['tokens']) for c in counts]
#     fnames = ['distr_chars_per_word_' + str(i + 1)  for i in range(max_chars)]
    return r
    
def character_count(entry):
    r = len(re.sub('\s+', '', entry['preprocessed']))
    return r


#https://github.com/ashenoy95/writeprints-static/blob/master/whiteprints-static.py
def hapax_legomena(entry):
    freq = nltk.FreqDist(word for word in entry['tokens'])
    hapax = [key for key, val in freq.items() if val == 1]
    dis = [key for key, val in freq.items() if val == 2]
    if len(dis) == 0 or len(entry['tokens']) == 0:
        return 0
    return (len(hapax) / len(dis)) / len(entry['tokens'])

def n_of_pronouns(entry):
    # Gab: aggiunto da noi
    return entry['pos_tags'].count('PRP') + entry['pos_tags'].count('PRP$')

def gunning_fog_index(entry):
    # Gab: aggiunto da noi
    r = readability.GunningFogIndex(entry['preprocessed'])
    gf = r.GunningFogIndex()
    return gf.score

def pass_fn(x):
    return x

class DependencyFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def get_feature_names_out(self):
        return ['num_tokens', 'max_parse_tree_depth', 'avg_parse_tree_depth']
    
    def transform(self, X):
        # X is assumed to be a list of dictionaries, each with a 'preprocessed' key for the text
        dep_features = []
        for entry in X:
            doc = nlp(str(entry['preprocessed']))  # Use the preprocessed text for parsing
            num_tokens = len(doc)
            depths = [self.get_depth(token) for token in doc]
            max_depth = max(depths) if depths else 0
            avg_depth = sum(depths) / len(depths) if depths else 0
            dep_features.append([num_tokens, max_depth, avg_depth])
        return dep_features

    @staticmethod
    def get_depth(token, depth=0):
        """
        Recursively gets the depth of a token within the dependency tree.
        A root token has a depth of 0.
        """
        if token.dep_ == 'ROOT':
            return depth
        else:
            return DependencyFeatures.get_depth(token.head, depth+1)

class Word2VecVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, word2vec_model):
        self.word2vec_model = word2vec_model
        self.vector_size = word2vec_model.vector_size

    def get_feature_names_out(self):
        return 'word2vec_'
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # X is assumed to be a list of dictionaries each containing 'tokens'
        return np.array([
            np.mean([self.word2vec_model[token] for token in entry['tokens'] if token in self.word2vec_model]
                    or [np.zeros(self.vector_size)], axis=0)
            for entry in X
        ])
    
    

#experiment = fe(load_corpus('pan13-train.jsonl'),load_corpus('pan13-test.jsonl'))
#experiment.run()
#load_corpus('pan20-train-small.jsonl',50)
experiment = fe(load_corpus('pan20-train-small.jsonl',5000),load_corpus('pan20-test.jsonl',5000))
experiment.run()