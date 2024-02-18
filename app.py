#!/usr/bin/python3
# encoding: utf-8

from settings.arguments import parse_arguments_controller
from module.precondition_checks import prechecks_func
from module.generate_pccd import generate_partial_contract_cheating_dataset
from module.load_json_file import read_jsonl
from module.measurement import time_function

def main():
    prechecks_func()
    generate_partial_contract_cheating_dataset()
    time_function(read_jsonl,'../preprocessed_datasets/pan20-train-small.jsonl',30000)

if __name__ == '__main__':
    main()