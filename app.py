#!/usr/bin/python3
# encoding: utf-8

from settings.arguments import parse_arguments_controller
from module.setup import setup
from module.generate_pccd import generate_partial_contract_cheating_dataset
from module.load_json_file import read_jsonl
from module.measurement import time_function

from settings.static_values import EXPECTED_PREPROCESSED_DATASETS_FOLDER, EXPECTED_PREPROCESSED_DATASET_FILES


def main():
    setup()

    #generate_partial_contract_cheating_dataset()
    #time_function(read_jsonl,EXPECTED_PREPROCESSED_DATASETS_FOLDER+EXPECTED_PREPROCESSED_DATASET_FILES['pan20-train-small'],30000)

if __name__ == '__main__':
    main()