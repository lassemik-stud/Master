import os

from module.restructure_db.restructure_pan13 import restructure_pan13_func
from module.restructure_db.restructure_pan20 import restructure_pan20_func
from module.restructure_db.restructure_pan23 import restructure_pan23_func
from settings.logging import print_l
from settings.expected_values import EXPECTED_DATASETS_FILES_PAN20, EXPECTED_DATASETS_FILES_PAN20, EXPECTED_DATASETS_FILES_PAN23, EXPECTED_PREPROCESSED_DATASET_FILES, EXPTECTED_DATASETS_FOLDERS_PAN13

def verify_correct_location_of_datasets(datasets_path="../datasets"):
    found_pan20_files_dict = {}
    found_pan13_dirs_dict = {}

    for root, dirs, files in os.walk(datasets_path):
        for file in files:
            if file in EXPECTED_DATASETS_FILES_PAN20:
                found_pan20_files_dict[file] = os.path.join(root, file)

        for dir in dirs:
            full_dir_path = os.path.join(root, dir)
            if dir in EXPTECTED_DATASETS_FOLDERS_PAN13 and os.path.isdir(full_dir_path):
                found_pan13_dirs_dict[dir] = full_dir_path

    # Check for missing files and directories
    missing_files = set(EXPECTED_DATASETS_FILES_PAN20) - found_pan20_files_dict.keys()
    missing_dirs = set(EXPTECTED_DATASETS_FOLDERS_PAN13) - found_pan13_dirs_dict.keys()

    # Print results
    for file in missing_files:
        print_l('ERROR',f"File missing: {file}")
    for dir in missing_dirs:
        print_l('ERROR',f"Directory missing: {dir}")

    if not missing_files and not missing_dirs:
        print_l("INFO", "Located all dataset files successfully")

    return found_pan20_files_dict, found_pan13_dirs_dict

def restructure_pan13_pre(pan13_dirs_dict):
    pan13_test_dir_path = pan13_dirs_dict['pan13-authorship-verification-test-corpus2-2013-05-29']
    pan13_test_truth_path = pan13_test_dir_path + "/truth.txt"
    pan13_test_output_path = "../preprocessed_datasets/pan13-test.jsonl"

    pan13_train_dir_path = pan13_dirs_dict['pan13-authorship-verification-training-corpus-2013-02-01']
    pan13_train_truth_path = pan13_train_dir_path + "/truth.txt"
    pan13_train_output_path = "../preprocessed_datasets/pan13-train.jsonl"

    restructure_pan13_func(pan13_test_dir_path, pan13_test_truth_path, pan13_test_output_path)
    restructure_pan13_func(pan13_train_dir_path, pan13_train_truth_path, pan13_train_output_path)

def restructure_pan20_pre(pan20_files_dict):
    pan20_train_small_pairs = pan20_files_dict["pan20-authorship-verification-training-small.jsonl"]
    pan20_train_small_truth = pan20_files_dict["pan20-authorship-verification-training-small-truth.jsonl"]
    pan20_train_small_output = "../preprocessed_datasets/pan20-train-small.jsonl"

    pan20_train_large_pairs = pan20_files_dict["pan20-authorship-verification-training-large.jsonl"]
    pan20_train_large_truth = pan20_files_dict["pan20-authorship-verification-training-large-truth.jsonl"]
    pan20_train_large_output = "../preprocessed_datasets/pan20-train-large.jsonl"

    pan20_test_pairs = pan20_files_dict["pan20-authorship-verification-test.jsonl"]
    pan20_test_truth = pan20_files_dict["pan20-authorship-verification-test-truth.jsonl"]
    pan20_test_output = "../preprocessed_datasets/pan20-test.jsonl"

    pan21_test_pairs = pan20_files_dict["pan21-authorship-verification-test.jsonl"]
    pan21_test_truth = pan20_files_dict["pan21-authorship-verification-test-truth.jsonl"]
    pan21_test_output = "../preprocessed_datasets/pan21-test.jsonl"

    restructure_pan20_func(pan20_train_small_pairs, pan20_train_small_truth, pan20_train_small_output)
    restructure_pan20_func(pan20_train_large_pairs, pan20_train_large_truth, pan20_train_large_output)
    restructure_pan20_func(pan20_test_pairs, pan20_test_truth, pan20_test_output)
    restructure_pan20_func(pan21_test_pairs, pan21_test_truth, pan21_test_output)

def check_for_preprocessed_datasets_files():
    found_preprocessed_files_dict = {}

    for root, dirs, files in os.walk("../preprocessed_datasets"):
        for file in files:
            if file in EXPECTED_PREPROCESSED_DATASET_FILES:
                found_preprocessed_files_dict[file] = os.path.join(root, file)

    # Check for missing files and directories
    missing_files = set(EXPECTED_PREPROCESSED_DATASET_FILES) - found_preprocessed_files_dict.keys()

    if not missing_files:
        print_l("INFO", "Loaded all dataset files successfully from preprocessed directory")
        return True
    else:
        print_l("ERROR", "Error in loading dataset files from preprocessed directory")
        return False

def prechecks_func():
    """
    Checks for correct dataset files.
    Structures the pan13, pan20 and pan23 to the same jsonl format. 
    """
    if check_for_preprocessed_datasets_files():
        print_l("DEBUG", "Preprocessed datasets already exist. Skipping prechecks.")
        # TRUE --> GOTO 
    else:
        pan20_files_dict, pan13_dirs_dict = verify_correct_location_of_datasets()
        restructure_pan13_pre(pan13_dirs_dict)
        restructure_pan20_pre(pan20_files_dict)
        check_for_preprocessed_datasets_files()

