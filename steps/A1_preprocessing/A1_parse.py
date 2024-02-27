# inspiration from https://github.com/boenninghoff/pan_2020_2021_authorship_verification/blob/main/preprocessing/step1_parse_and_split.py

import json

from settings.logging import printLog as printLog

class Problem():
    def __init__(self):
        self.id = 0
        self.dataset = 0
        self.type = ""
        self.author = ""
        self.same_author = 0
        self.known_text = set()
        self.problem_text = set()
        self.known_retrieval_location = set()
        self.uknown_put_location = set()
        self.additional_info = ""
    
    def __str__(self):
        return f"ID: {self.id}, Dataset: {self.dataset}, Type: {self.type}, Author: {self.author}, Same Author: {self.same_author}, Known Text: {self.known_text}, Problem Text: {self.problem_text}, Known Retrieval Location: {self.known_retrieval_location}, Unknown Put Location: {self.uknown_put_location}, Additional Info: {self.additional_info}"

class Corpus():
    def __init__(self):
        
        # split set into training, validation and calibration
        self.all = set()
        self.train = set()
        self.val = set()
        self.cal = set()

        # unique authors, types and documents
        self.authors = set()
        self.types = set()
        self.unique_docs = set()

        # counting calues
        self.n_train = 0
        self.n_cal = 0
        self.n_val = 0 
        self.n_dropped = 0

        # Statistics 
        self.avg_number_of_words = 0
        self.avg_number_of_characters = 0

    # Parse the raw data from the json file
    def parse_raw_data(self,json_file):
        with open(json_file, 'r',encoding='utf-8') as file:
            lines = file.readlines()
        
        for line in lines: 
            entry = json.loads(line)
            problem = Problem()
            problem.id = entry.get('id')
            problem.dataset = entry.get('dataset')
            problem.type = entry.get('type')
            problem.author = entry.get('author')
            problem.same_author = entry.get('same author')
            problem.known_text = entry.get('known text')
            problem.problem_text = entry.get('problem_text')
            problem.known_retrieval_location = entry.get('_known_text_put_location')
            problem.uknown_put_location = entry.get('_unknown_text_retrieval_location')
            problem.additional_info = entry.get('additional_info')
            
            self.unique_docs.add(problem.id)
            self.authors.add(str(problem.author))
            self.types.add(str(problem.type))
            self.all.add(problem)
        
    # Split the corpus into training, validation and calibration sets
    def split_corpus(self,train_size,val_size):
        self.n_train = int(len(self.all)*train_size)
        self.n_val = int(len(self.all)*val_size)
        self.n_cal = len(self.all) - self.n_train - self.n_val
        self.n_dropped = len(self.all) - self.n_train - self.n_val - self.n_cal

        self.train = set(list(self.all)[:self.n_train])
        self.val = set(list(self.all)[self.n_train:self.n_train+self.n_val])
        self.cal = set(list(self.all)[self.n_train+self.n_val:])

    # Get the average statistics of the corpus
    def get_avg_statistics(self):
        entry_count = 0
        for problem in self.all:
            for entry in problem.known_text:
                entry_count += 1
                for sentence in entry:
                    self.avg_number_of_words +=len(sentence.split(" "))
                    for character in sentence:
                        self.avg_number_of_characters += 1
            
            entry_count += 1
            for sentence in problem.problem_text:
                self.avg_number_of_words += len(sentence.split(" "))
                for character in sentence:
                    self.avg_number_of_characters += 1

        self.avg_number_of_words = self.avg_number_of_words/entry_count
        self.avg_number_of_characters = self.avg_number_of_characters/entry_count

    def print_corpus_info(self):
        printLog.debug(f"Number of documents in training set: {self.n_train}")
        printLog.debug(f"Number of documents in validation set: {self.n_val}")
        printLog.debug(f"Number of documents in calibration set: {self.n_cal}")
        printLog.debug(f"Number of documents dropped: {self.n_dropped}")
        printLog.debug(f"Number of unique authors: {len(self.authors)}")
        printLog.debug(f"Number of unique types: {len(self.types)}")
        printLog.debug(f"Number of unique documents: {len(self.unique_docs)}")
        printLog.info(f"Average number of words in a document: {self.avg_number_of_words}")
        printLog.info(f"Average number of characters in a document: {self.avg_number_of_characters}")
        
    
