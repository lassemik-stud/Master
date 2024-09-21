import spacy
import numpy as np
import torch
# from nltk.util import ngrams
from transformers import AutoTokenizer, AutoModel


# Load spaCy's English language model
nlp = spacy.load('en_core_web_sm')

def sentences(text, n):
    sents = [i.text for i in nlp(text).sents]
    return [' '.join(sents[i:i+n]) for i in range(0, len(sents), n)]

def calculate_word_length_distribution(text):
    """
    Calculate the normalized distribution of word lengths in a text.
    """
    word_lengths = [len(token.text) for token in nlp(text)]
    bins = [1, 4, 7, 10, 13, np.inf]  # Define bins for word lengths
    hist, _ = np.histogram(word_lengths, bins=bins)  # Calculate histogram
    distribution = hist / sum(hist) if sum(hist) > 0 else hist  # Normalize histogram
    return distribution.tolist()  # Convert to list for easier handling

def calculate_vocabulary_richness(text):
    """
    Calculate the Type-Token Ratio (TTR) as a measure of vocabulary richness.
    TTR is the ratio of unique word types to the total number of words.
    """
    doc = nlp(text)
    # Exclude punctuation and whitespace for TTR calculation
    words = [token.text for token in doc if not token.is_punct and not token.is_space]
    types = set(words)
    if len(words) == 0:  # Avoid division by zero
        return 0
    ttr = len(types) / len(words)
    return ttr

def get_mbert_embedding(text, tokenizer, model):
    """
    Generate an M-BERT embedding for the input text.
    """
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state
    pooled_embedding = torch.mean(embeddings, dim=1)
    return pooled_embedding[0].numpy()

def spacy_tokenizer(arg):
    pair = arg[0]
    arguments = arg[1]

    feature_type = arguments.get('feature_type')
    #special_chars = arguments.get('special_chars')
    word_length_dist = arguments.get('word_length_dist')
    include_vocab_richness = arguments.get('include_vocab_richness')

    tfidf_bow_flag = True if feature_type in ['tfidf', 'BoW'] else False
    embedding_flag = True if feature_type == 'word_embeddings' else False
    dependencies_flag = True if feature_type == 'dependency' else False
    bert_m_flag = True if feature_type == 'bert_m' else False
    #pos_tags_flag = True if feature_type == 'dependency' else False
    word_length_dist_flag = True if word_length_dist else False
    vocab_richness_flag = True if include_vocab_richness else False
    #special_chars_flag = True if special_chars else False

    kt = pair[0]
    ut = pair[1]

    # Tokenize the texts
    doc_kt = nlp(kt)
    doc_ut = nlp(ut)

    # Calculate word length distributions for each text
    word_length_dist_kt = calculate_word_length_distribution(kt) if word_length_dist_flag else 0
    word_length_dist_ut = calculate_word_length_distribution(ut) if word_length_dist_flag else 0

    # Calculate vocabulary richness for each text
    vocab_richness_kt = calculate_vocabulary_richness(kt) if vocab_richness_flag else 0
    vocab_richness_ut = calculate_vocabulary_richness(ut) if vocab_richness_flag else 0

    # Embeddings - Word embeddings with spaCy
    embedding_kt = np.mean([token.vector for token in doc_kt if not token.is_punct and not token.is_space], axis=0) if embedding_flag else 0
    embedding_ut = np.mean([token.vector for token in doc_ut if not token.is_punct and not token.is_space], axis=0) if embedding_flag else 0

    # BERT-M - embeddings 
    if bert_m_flag:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
        model = AutoModel.from_pretrained("bert-base-multilingual-cased")
        bert_m_kt = get_mbert_embedding(kt, tokenizer, model)
        bert_m_ut = get_mbert_embedding(ut, tokenizer, model)
    else:
        bert_m_kt = bert_m_ut = 0

    kt_token = {
        'dependencies':     [token.dep_ for token in doc_kt] if dependencies_flag else 0,
        'pos':              [token.pos_ for token in doc_kt] if dependencies_flag else 0,
        'lemma':            " ".join([token.lemma_ for token in doc_kt]) if tfidf_bow_flag else 0,
        'word_length_dist': word_length_dist_kt,
        'vocab_richness':   vocab_richness_kt,
        'embedding':        embedding_kt,
        'bert_m' :          bert_m_kt
    }

    ut_token = {
        'dependencies':     [token.dep_ for token in doc_ut] if dependencies_flag else 0,
        'pos':              [token.pos_ for token in doc_ut] if dependencies_flag else 0,
        'lemma':            " ".join([token.lemma_ for token in doc_ut]) if tfidf_bow_flag else 0,
        'word_length_dist': word_length_dist_ut,
        'vocab_richness':   vocab_richness_ut,
        'embedding':        embedding_ut,
        'bert_m' :          bert_m_ut  
    }

    return [kt_token, ut_token]
