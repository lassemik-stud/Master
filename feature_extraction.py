
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np

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