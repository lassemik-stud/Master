from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
from collections import Counter
import spacy
from settings.logging import printLog

from sklearn.metrics.pairwise import cosine_similarity


# Initialize spaCy
nlp = spacy.load("en_core_web_sm")

class TextPairFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, ngram_range=(1,1), max_features=None, feature_type='tfidf', analyzer='word', special_chars=False, word_length_dist=False, include_vocab_richness=False):
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.feature_type = feature_type
        self.analyzer = analyzer
        self.special_chars = special_chars
        self.word_length_dist = word_length_dist
        self.include_vocab_richness = include_vocab_richness
        
        # Initialize with the specified feature extractor
        if self.feature_type in ['tfidf', 'BoW']:
            if special_chars:
                # Define a regex pattern for special characters
                # This pattern matches characters that are not alphanumeric (and not spaces)
                token_pattern = r'(?u)[^\w\s]'
                self.analyzer = 'char_wb'  # Analyze at the character level within word boundaries
                
            vectorizer_class = TfidfVectorizer if self.feature_type == 'tfidf' else CountVectorizer
            self.vectorizer = vectorizer_class(tokenizer=None, ngram_range=self.ngram_range, max_features=self.max_features, analyzer=self.analyzer)
        elif self.feature_type == 'dependency':
            # Define a fixed list of dependency and POS tags you expect to see
            self.feature_tags = ['nsubj', 'dobj', 'ROOT', 'NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'DET', 'ADP', 'NUM', 'CONJ', 'PUNCT', 'PART', 'SCONJ', 'SYM', 'X', 'INTJ',]
            self.vectorizer = None  # Not using a traditional vectorizer for dependency features   
        elif self.feature_type == 'word_embeddings':
            self.vectorizer = None 
        else:
            printLog.error(f"Unsupported feature type: {self.feature_type}")
            raise ValueError("Unsupported feature type. Choose 'tfidf', 'BoW' or 'dependency'.")

    def fit(self, X, y=None):
        if self.feature_type in ['tfidf', 'BoW']:
            # Extract all texts based on the 'lemma' key from the input dictionaries
            all_texts = [text['lemma'] for pair in X for text in pair]
            self.vectorizer.fit(all_texts)
        
        return self

    def transform(self, X, y=None):
        features = []
        for pair in X:
            feature_vecs = []
            if self.feature_type in ['tfidf', 'BoW']:
                text1, text2 = pair[0]['lemma'], pair[1]['lemma']
                feat1 = self.vectorizer.transform([text1]).toarray()
                feat2 = self.vectorizer.transform([text2]).toarray()
                feat_diff = np.abs(feat1 - feat2)
                feature_vecs.append(feat_diff.flatten())

            elif self.feature_type == 'dependency':
                # Extract and vectorize dependency and POS tag features for both texts
                dep_pos_features1 = self._vectorize_dep_pos_features(pair[0])
                dep_pos_features2 = self._vectorize_dep_pos_features(pair[1])
                # Compute the difference between the two feature vectors
                feat_diff = np.abs(dep_pos_features1 - dep_pos_features2)
                feature_vecs.append(feat_diff)
            elif self.feature_type == 'word_embeddings':
                emb1 = pair[0]['embedding'].reshape(1, -1)
                emb2 = pair[1]['embedding'].reshape(1, -1)
                
                # Compute cosine similarity, result is in a 2D array
                cos_sim = cosine_similarity(emb1, emb2)
                cos_sim_value = cos_sim[0, 0]
                cos_dis = 1 - cos_sim_value
                feature_vecs.append(np.array([cos_dis]).flatten())

                #emb_diff = np.linalg.norm(pair[0]['embedding'] - pair[1]['embedding'])
                #feature_vecs.append(np.array([emb_diff])) 
            if self.word_length_dist:
                wld_diff = np.abs(np.array(pair[0]['word_length_dist']) - np.array(pair[1]['word_length_dist']))
                feature_vecs.append(wld_diff.flatten())
            if self.include_vocab_richness:
                vocab_richness_diff = np.abs(np.array(pair[0]['vocab_richness']) - np.array(pair[1]['vocab_richness']))
                feature_vecs.append(vocab_richness_diff.flatten())
            
            combined_features = np.concatenate(feature_vecs, axis=None)
            features.append(combined_features.flatten())

        return np.array(features)

    def _vectorize_dep_pos_features(self, text_data):
        """Converts dependency and POS tag data into a fixed-order feature vector."""
        combined_features = text_data['dependencies'] + text_data['pos']
        feature_counts = Counter(combined_features)
        # Create a feature vector based on the fixed list of tags, ensuring a consistent order
        feature_vector = np.array([feature_counts.get(tag, 0) for tag in self.feature_tags])
        return feature_vector