# inspiration from https://github.com/boenninghoff/pan_2020_2021_authorship_verification/blob/main/preprocessing/step1_parse_and_split.py

import json

from settings.logging import printLog as printLog
from sklearn.utils import shuffle 
import random 


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

class Corpus():
    def __init__(self):
        
        # split set into training, validation and calibration
        self.all = []
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

        # Number of same author and different author pairs
        self.n_same_author = 0
        self.diff_author = 0

        # Statistics 
        self.avg_number_of_words = 0
        self.avg_number_of_characters = 0

    # Parse the raw data from the json file
    def parse_raw_data(self,json_file):
        printLog.debug(f"Reading data from {json_file}")
        with open(json_file, 'r',encoding='utf-8') as file:
            lines = file.readlines()
        
        for line in lines: 
            entry = json.loads(line)
            instance = Instance()
            instance.id = entry.get('id')
            instance.dataset = entry.get('dataset')
            instance.type = entry.get('type')
            instance.author = entry.get('author')
            instance.same_author = entry.get('same author')
            instance.known_text = entry.get('known text')
            instance.unknown_text = entry.get('unknown text')
            instance.additional_info = entry.get('additional_info')
            self.unique_docs.add(instance.id)
            self.authors.add(str(instance.author))
            self.types.add(str(instance.type))
            self.all.append(instance)
        

    # Split the corpus into training, validation and calibration sets
    def split_corpus(self,train_size,val_size):
        self.n_train = int(len(self.all)*train_size)
        self.n_val = int(len(self.all)*val_size)
        self.n_cal = len(self.all) - self.n_train - self.n_val
        self.n_dropped = len(self.all) - self.n_train - self.n_val - self.n_cal

        self.train = set(list(self.all)[:self.n_train])
        self.val = set(list(self.all)[self.n_train:self.n_train+self.n_val])
        self.cal = set(list(self.all)[self.n_train+self.n_val:])
    
    def select_balanced_corpus(self, max_instances):
        """
        Selects a balanced corpus with an equal number of instances having the same and different authors.

        :param corpus: List of instances (assumed to have 'same_author' attribute).
        :param x: Total number of instances to select.
        :return: A list of instances with 50% having the same author and 50% having different authors.
        """
        # Ensure x is even to divide evenly between same and different authors
        if max_instances % 2 != 0:
            raise ValueError("The number of entries 'x' must be even to balance same and different authors.")

        # Shuffle the corpus to ensure randomness
        random.shuffle(self.all)

        # Partition the corpus into same_author and different_author
        same_author_instances = [instance for instance in self.all if instance.same_author == 1]
        different_author_instances = [instance for instance in self.all if instance.same_author != 1]

        # Determine the number of entries from each partition
        num_each = max_instances // 2

        # Randomly select instances from each partition
        selected_same_author = random.sample(same_author_instances, num_each)
        selected_different_author = random.sample(different_author_instances, min(num_each, len(different_author_instances)))

        # Combine and shuffle the selected instances
        balanced_corpus = selected_same_author + selected_different_author
        random.shuffle(balanced_corpus)

        self.all = balanced_corpus

    # Get the average statistics of the corpus
    def get_avg_statistics(self):
        entry_count = 0
        for instance in self.all:
            if int(instance.same_author) == 1:
                self.n_same_author += 1
            else:
                self.diff_author += 1

            # Average number of words and characters
            entry_count += 1
            self.avg_number_of_words +=len(instance.known_text)
            for character in instance.known_text:
              self.avg_number_of_characters += 1
            
            entry_count += 1
            self.avg_number_of_words += len(instance.unknown_text.split(" "))
            for character in instance.unknown_text:
                self.avg_number_of_characters += 1

        self.avg_number_of_words = self.avg_number_of_words/entry_count
        self.avg_number_of_characters = self.avg_number_of_characters/entry_count

    def print_corpus_info(self):
        printLog.debug(f"Number of documents in set: {len(self.all)}")
        printLog.debug(f"Number of same author (SA) in set: {self.n_same_author}")
        printLog.debug(f"Number of different author (DA) in set: {self.diff_author}")
        #printLog.debug(f"Number of documents in validation set: {self.n_val}")
        #printLog.debug(f"Number of documents in calibration set: {self.n_cal}")
        printLog.debug(f"Number of documents dropped: {self.n_dropped}")
        printLog.debug(f"Number of unique authors: {len(self.authors)}")
        printLog.debug(f"Number of unique types: {len(self.types)}")
        printLog.debug(f"Number of unique documents: {len(self.unique_docs)}")
        printLog.info(f"Average number of words in a document: {self.avg_number_of_words}")
        printLog.info(f"Average number of characters in a document: {self.avg_number_of_characters}")
        
    
