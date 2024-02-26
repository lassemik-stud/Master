#!/usr/bin/python3
# encoding: utf-8

from settings.arguments import parse_arguments_controller
from steps.setup import setup
from steps.load_json_file import read_jsonl

from settings.static_values import EXPECTED_PREPROCESSED_DATASETS_FOLDER, EXPECTED_PREPROCESSED_DATASET_FILES


def main():
    setup()

if __name__ == '__main__':
    main()