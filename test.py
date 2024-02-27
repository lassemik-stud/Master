
import os

from steps.A1_preprocessing.A1_parse import Corpus

from settings.static_values import EXPECTED_PREPROCESSED_DATASETS_FOLDER as EXP_PRE_DATASETS_FOLDER
from steps.setup import setup


setup()

Corpus = Corpus()
Corpus.parse_raw_data(os.path.join(EXP_PRE_DATASETS_FOLDER,"pan13-partial-contract-cheating-test.jsonl"))
Corpus.split_corpus(0.7,0.15)
Corpus.get_avg_statistics()
Corpus.print_corpus_info()