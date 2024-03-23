
# Bag of words + svm 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm

# Word Vectors
import spacy

# Stemming / Lemmatization
import nltk

# CountVectorizer - a type of Bag of Words 
 

# UNIGRAM APROACH
class Category: 
    BOOKS = "BOOKS"
    CLOTHES = "CLOTHES"

train_x = ["i love the book book", "this is a great book", "the fit is great", "i love the shoes"]
train_y = [Category.BOOKS, Category.BOOKS, Category.CLOTHES, Category.CLOTHES]

vectorizer = CountVectorizer(binary=True)
train_x_vectors = vectorizer.fit_transform(train_x)

clf_svm = svm.SVC(kernel='linear')
clf_svm.fit(train_x_vectors, train_y)

test_x = vectorizer.transform(['shoes are alright'])

clf_svm.predict(test_x)

# BIGRAM APROACH
class Category: 
    BOOKS = "BOOKS"
    CLOTHES = "CLOTHES"

train_x = ["i love the book book", "this is a great book", "the fit is great", "i love the shoes"]
train_y = [Category.BOOKS, Category.BOOKS, Category.CLOTHES, Category.CLOTHES]

vectorizer = CountVectorizer(binary=True, ngram_range=(1,2))
train_x_vectors = vectorizer.fit_transform(train_x)

# Word Vectors - capture the semantic meaning of a word in a vector. Capture the context
# Word embeddings
nlp = spacy.load('en_core_web_md')
train_x = ["i love the book", "this is a great book", "the fit is great", "i love the shoes"]


docs = [nlp(text) for text in train_x]
train_x_word_vectors = [x.vector for x in docs]

clf_svm_wv = svm.SVC(kernel='linear')
clf_svm_wv.fit(train_x_word_vectors, train_y)

test_x = ["i went to the bank and wrote a check", "let me check that out"]
test_docs = [nlp(text) for text in test_x]
test_x_word_vectors = [x.vector for x in test_docs]

print(clf_svm_wv.predict(test_x_word_vectors))

# Stemming / Lemmatization - reduce words to their root form
# Stories -> Story (lemmatization). Stories -> Stori (stemming)
# Lemmatization is more accurate but slower
# Stemming is faster but less accurate
#nltk.download('wordnet')
#nltk.download('stopwords')
#nltk.download('punkt')

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
phrase = "reading the books and the stories"
words_tokenized = word_tokenize(phrase)

stemmed_words = []
for word in words_tokenized:
    stemmed_words.append(stemmer.stem(word))
print(" ".join(stemmed_words))

# Lemmatization 
# should do part of speech tagging to utilized lemmatization better
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
lemmatized_words = []
for word in words_tokenized:
    lemmatized_words.append(lemmatizer.lemmatize(word))
print(" ".join(lemmatized_words))

# Stopwords removal 
from nltk.corpus import stopwords

_stopwords = stopwords.words('english')
phrase = "Here is an example sentence that is demonstrating the removal of stop words"
words_tokenized = word_tokenize(phrase)

stripped_phrase = []
for word in words_tokenized: 
    if word not in _stopwords:
        stripped_phrase.append(word)

print(" ".join(stripped_phrase))

# Various other techniques. Spell correction, sentimen, & pos tagging

# Spell correction
from textblob import TextBlob
phrase = "this is an example of a sentance with a speling error"

blob = TextBlob(phrase)
print(blob.correct())

# Tag with Blob
# py -m textblob.download_corpora
phrase = "the book was really horrible"
blob = TextBlob(phrase)
print(blob.tags)

# Tag with nltk
text = word_tokenize(phrase)
print(nltk.pos_tag(text))

# Sentiment analysis
print(blob.sentiment)

# Recurrent Neural Networks 
# BERT 
# py -m pip install spacy-transformers 
# py -m spacy download en_trf_bertbaseuncased_lg

