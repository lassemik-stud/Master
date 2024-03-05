# https://medium.com/@eskandar.sahel/exploring-feature-extraction-techniques-for-natural-language-processing-46052ee6514
# https://www.analyticsvidhya.com/blog/2022/05/a-complete-guide-on-feature-extraction-techniques/

import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF, PCA
from sklearn.manifold import TSNE
import numpy as np
import spacy

class extractFeatures():
    def __init__(self):
        self.corpus = set()
    
    def tokenize(self, text):
        return word_tokenize("".join(text))
    
    def bag_of_words(self):
        return {word: self.tokens.count(word) for word in set(self.tokens)}

    def load_corpus(self,corpus):
        self.corpus = corpus
    
    def extract_features(self):
        for problem in self.corpus:
            problem.token_kt = self.tokenize(problem.known_text)
            problem.bow_kt = self.bag_of_words()

            problem.token_pt = self.tokenize(problem.problem_text)
            problem.bow_pw = self.bag_of_words()
            
