import os

from steps.A0_prepare_datasets.A0_restructure_pan13 import restructure_pan13
from steps.A0_prepare_datasets.A2_restructure_pan20 import restructure_pan20
from steps.A0_prepare_datasets.A3_restructure_pan23 import restructure_pan23
from settings.logging import printLog as PrintLog
from settings.static_values import EXPECTED_DATASETS_FILES_PAN20, EXPECTED_DATASETS_FILES_PAN20, EXPECTED_DATASETS_FILES_PAN23, EXPECTED_PREPROCESSED_DATASET_FILES, EXPECTED_DATASETS_FOLDERS_PAN13, EXPECTED_DATASETS_FOLDER, EXPECTED_PREPROCESSED_DATASETS_FOLDER
from steps.A0_prepare_datasets.A1_create_problem_pan13 import create_pan13_problem

import os
import zipfile

def unzip_all(directory):
    # Find all zip files in the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".zip"):
                # Construct the full file path
                file_path = os.path.join(root, file)
                # Open the zip file
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    # Extract all the contents of the zip file in current directory
                    zip_ref.extractall(root)
                # Remove the zip file
                os.remove(file_path)
    # Check if there are any new zip files in the directory
    new_zip_files = any(file.endswith(".zip") for root, dirs, files in os.walk(directory) for file in files)
    if new_zip_files:
        # If there are new zip files, call the function recursively
        unzip_all(directory)

def verify_correct_location_of_datasets(datasets_path=EXPECTED_DATASETS_FOLDER):
    identified_pan_20_files_dict = {}
    identified_pan_13_files_dict = {}

    for root, dirs, files in os.walk(datasets_path):
        for file in files:
            if file in EXPECTED_DATASETS_FILES_PAN20.values():
                identified_pan_20_files_dict[file] = os.path.join(root, file)

        for dir in dirs:
            full_dir_path = os.path.join(root, dir)
            if dir in EXPECTED_DATASETS_FOLDERS_PAN13.values() and os.path.isdir(full_dir_path):
                # Check if there is a subdirectory with the same name
                subdir_path = os.path.join(full_dir_path, dir)
                if os.path.isdir(subdir_path):
                    # If the subdirectory exists, use its path
                    identified_pan_13_files_dict[dir] = subdir_path
                else:
                    # Otherwise, use the original directory path
                    identified_pan_13_files_dict[dir] = full_dir_path

    missing_files = set(EXPECTED_DATASETS_FILES_PAN20.values()) - identified_pan_20_files_dict.keys()
    missing_dirs = set(EXPECTED_DATASETS_FOLDERS_PAN13.values()) - identified_pan_13_files_dict.keys()

    # Print results
    for file in missing_files:
        PrintLog.error(f"File missing: {file}")
    for dir in missing_dirs:
        PrintLog.error(f"Directory missing: {dir}")
    if not missing_files and not missing_dirs:
        PrintLog.info(f"Located all dataset files successfully")

    return identified_pan_20_files_dict, identified_pan_13_files_dict

def restructure_pan13_pre(pan13_dirs_dict):
    for dataset in ['pan13-test', 'pan13-train']:
        restructure_pan13(main_directory=pan13_dirs_dict[EXPECTED_DATASETS_FOLDERS_PAN13[dataset]], 
                            truth_txt_path=pan13_dirs_dict[EXPECTED_DATASETS_FOLDERS_PAN13[dataset]] + "\\truth.txt", 
                            output_file=EXPECTED_PREPROCESSED_DATASETS_FOLDER+EXPECTED_PREPROCESSED_DATASET_FILES[dataset])

def restructure_pan20_pre(pan20_files_dict):
    for dataset in  ['pan20-train-small','pan20-train-large','pan20-test','pan21-test']:
        restructure_pan20(path_to_pairs=pan20_files_dict[EXPECTED_DATASETS_FILES_PAN20[dataset]],
                               path_to_truth=pan20_files_dict[EXPECTED_DATASETS_FILES_PAN20[dataset+"-truth"]],
                               output_path=EXPECTED_PREPROCESSED_DATASETS_FOLDER+EXPECTED_PREPROCESSED_DATASET_FILES[dataset])

def check_for_preprocessed_datasets_files():
    found_preprocessed_files_dict = {}
    for root, dirs, files in os.walk(EXPECTED_PREPROCESSED_DATASETS_FOLDER):
        for file in files:
            if file in EXPECTED_PREPROCESSED_DATASET_FILES.values():
                found_preprocessed_files_dict[file] = os.path.join(root, file)

    # Check for missing files and directories
    missing_files = set(EXPECTED_PREPROCESSED_DATASET_FILES.values()) - found_preprocessed_files_dict.keys()
    for file in missing_files:
        PrintLog.error(f"File missing: {file}")
    for file in found_preprocessed_files_dict.keys():
        PrintLog.info(f"File found: {file}")
    if not missing_files:
        PrintLog.info( "Loaded all dataset files successfully from preprocessed directory")
        return True
    else:
        PrintLog.error( "Error in loading dataset files from preprocessed directory")
        return False

def setup():
    if check_for_preprocessed_datasets_files():
        PrintLog.debug("Preprocessed datasets already exist. Skipping prechecks.")
        # TRUE --> GOTO 
    else:
        #unzip_all(EXPECTED_DATASETS_FOLDER)
        pan20_files_dict, pan13_dirs_dict = verify_correct_location_of_datasets()
        restructure_pan13_pre(pan13_dirs_dict)
        #restructure_pan20_pre(pan20_files_dict)
        check_for_preprocessed_datasets_files()
        create_pan13_problem(EXPECTED_PREPROCESSED_DATASETS_FOLDER+EXPECTED_PREPROCESSED_DATASET_FILES['pan13-train'])
        #print(length)

