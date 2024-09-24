# python3

# Function to simplify n-gram range
def simplify_ngram_range(ngram_range):
    if ngram_range == [1, 1]:
        return 1
    elif ngram_range == [2, 2]:
        return 2
    elif ngram_range == [3, 3]:
        return 3
    elif ngram_range == [4, 4]:
        return 4
    elif ngram_range == [5, 5]:
        return 5
    elif ngram_range == [1, 2]:
        return 6
    elif ngram_range == [2, 3]:
        return 7
    elif ngram_range == [3, 4]:
        return 8
    elif ngram_range == [4, 5]:
        return 9
    else:
        print(f"Unexpected ngram_range: {ngram_range}")
        return None  # or raise an exception

# Function to map feature type to color
def map_feature_type(feature_type):
    if feature_type == 'tfidf':
        return 'blue'
    elif feature_type == 'BoW':  # Assuming 'BoW' is the other feature type
        return 'red'
    else:
        print(feature_type)
        return 'gray'  # Default color for unexpected types

# Function to map feature analyzer to numerical values
def map_feature_analyzer(feature_analyzer):
    if feature_analyzer == 'word':
        return 2
    elif feature_analyzer == 'char':
        return 4
    elif feature_analyzer == 'char_wb':
        return 6
    return 0 

# Function to calculate extra feature
def map_special_char_and_inc_vocab_richness(special_char, include_vocab_richness):
    if [special_char,include_vocab_richness] == [True, True]:
        return 6
    elif [special_char,include_vocab_richness] == [True, False]:
        return 2
    elif [special_char,include_vocab_richness] == [False, True]:
        return 4
    elif [special_char,include_vocab_richness] == [False, False]:
        return 0
    else: 
        print(f"unexpected value - [{special_char}, {include_vocab_richness}]")
        return 0
    

def simplify_svm_c(value):
    if value == 0.01:
        return 1
    elif value == 0.1:
        return 2
    elif value == 1:
        return 3
    elif value == 10:
        return 4
    else:
        print(f"Unexpected value: {value}")
        return 0

def simplify_svm_kernel(value):
    if value == 'linear':
        return 1
    elif value == 'poly':
        return 2
    elif value == 'rbf':
        return 3
    elif value == 'sigmoid':
        return 4
    else:
        print(f"Unexpected value: {value}")
        return 0

def simplify_lr_c(value):
    if value == 0.01:
        return 1
    elif value == 0.1:
        return 2
    elif value == 1:
        return 3
    elif value == 10:
        return 4
    else:
        print(f"Unexpected value: {value}")
        return 0
    
def simplify_nb_alpha(value):
    if value == 0.01:
        return 1
    elif value == 0.1:
        return 2
    elif value == 1:
        return 3
    elif value == 10:
        return 4
    else:
        print(f"Unexpected value: {value}")
        return 0

def simplify_nb_fit_prior(value):
    if value == True:
        return 1
    elif value == False:
        return 2
    else:
        print(f"Unexpected value: {value}")
        return 0