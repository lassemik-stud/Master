# Status of coding 

## Overall steps 
- [ ]  Prepare datasets 
 - [ ] PAN 13
 - [ ] PAN 20
 - [ ] PAN 23
- [ ] Preprocessing 
 - [ ] Feature extraction from text 
- [ ] Feature selection
- [ ] Feature extraction (database terminology)
- [ ] Classification. 
- [ ] Data Evaluation 

### Prepare datasets 
- [ ] Standardize the datasets into a common json format
- [ ] PAN 13 dataset - create a partial contract cheating problem 

### Preprocessing 
The goal is to convert data into feature space. Scale features, handle missing data. 
In stylometric terminology this is "feature extraction". In ML it is just part of preprocessing.  
- [ ] 
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