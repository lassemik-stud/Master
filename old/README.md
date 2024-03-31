# Status of coding 

# GOAL UNTIL WEDNESDAY
- [ ] Create a prototype for partial contract cheating

## HOW
- [ ] MONDAY
Use the PAN 13 dataset. Do a json transformation. Do not add extra splittings. 
Perform some simple feature extraction. 
Perform a one class classifier on the data. 
- [ ] THURSDAY
Implement the rolling attribution method on the data above. 

- [ ] WEDNESDAY
Implement other feature extraction and classification methods. 

## Overall steps 
- [X]  Prepare datasets 
 - [X] PAN 13
 - [ ] PAN 20
 - [ ] PAN 23
- [ ] Preprocessing 
 - [ ] Feature extraction from text 
- [ ] Feature selection
- [ ] Feature extraction (database terminology)
- [ ] Classification. 
- [ ] Data Evaluation 

### Prepare datasets 
- [X] Standardize the datasets into a common json format
- [X] PAN 13 dataset - create a partial contract cheating problem 

### Preprocessing 
The goal is to convert data into feature space. Scale features, handle missing data. 
In stylometric terminology this is "feature extraction". In ML it is just part of preprocessing.  
- [X] Load dataset into Corpus class
- [X] Split dataset into TRAIN, VALIDATION and CALIBRATION SET
- [ ] Feature Extraction 
    - [ ] Syntactic Features. Syntax tree 
    - [ ] Lexial Features. 
        - [ ] Word based. Word n-grams.
        - [ ] Character based. Character n-grams.
    - [ ] Syntactic Features
        - [ ] Syntax tree. 
    - [ ] Semantic Features. Synonyms and semantic dependencies. 
    - [ ] Structural Features. 
    - [ ] Versification Features

    - [ ] Classical features 
        - [ ] N-grams
        - [ ] length of sentences
        - [ ] length of paragraphs and words
        - [ ] punctiantion marks
        - [ ] stop words and POS-tags
        - [ ] word and character embeddings
        - [ ] word and character based transformers
        - [ ] n-grams Bag og Words
        - [ ] unmasking
        - [ ] syntax trees 
    - [ ] Bag of words
- [ ] Classification
    - [ ] SVM
    - [ ] Random Forest
    - [ ] RNN
    - [ ] Logic Regression

- [ ] Identify python libraries for the task
**Libraries**
- Natural Language Toolkit (nltk) 'pip install nltk'
- spaCy. Another natural language processing kit 'pip install -U spacy' spacy.io. Tokenizer.
- 

**Pre trained models**
- word embeddings --> fasttext.cc --> English word vectors and Multi-lingual word vectors

# Best Practice
    - Use proper naming conventions for variables, functions, methods, and more.
    - Variables, functions, methods, packages, modules: this_is_a_variable
    - Classes and exceptions: CapWords
    - Protected methods and internal functions: _single_leading_underscore
    - Private methods: __double_leading_underscore
    - Constants: CAPS_WITH_UNDERSCORES
    - Use 4 spaces for indentation. For more conventions, refer to PEP8.