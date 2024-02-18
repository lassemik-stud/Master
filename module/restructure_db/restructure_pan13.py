import uuid
import json
import os
import glob
import sys

from settings.logging import print_l

def read_file_content(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().strip()

def generate_json_structure(directory, truth_file, dataset_number):
    # Load truth data
    with open(truth_file, 'r', encoding='utf-8') as file:
        truth_data = file.readlines()
    truth_dict = {line.split()[0]: (1 if 'Y' in line.split()[1] else 0) for line in truth_data}

    # Initialize the list to store all json entries
    json_data = []

    # Iterate over each subdirectory in the main directory
    for subdir in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, subdir)):
            # Initialize the entry dictionary
            entry = {
                "id": str(uuid.uuid4()),
                "dataset": dataset_number,
                "type": [subdir[:2]],  # Assuming first two characters denote the type
                "author": [],
                "same author": truth_dict.get(subdir, 0),
                "known text": [],
                "unknown text": "",
                "additional_info": {"subdirectory": subdir}
            }

            # Add known texts
            known_files = glob.glob(os.path.join(directory, subdir, 'known*.txt'))
            for file_path in known_files:
                entry["known text"].append(read_file_content(file_path).split(". "))

            # Add unknown text
            unknown_file_path = os.path.join(directory, subdir, 'unknown.txt')
            if os.path.exists(unknown_file_path):
                entry["unknown text"] = read_file_content(unknown_file_path).split(". ")

            # Add the entry to the list
            json_data.append(entry)

    return json_data

def restructure_pan13_func(main_directory, truth_txt_path, output_file):
    dataset_num = 13

    # Generate the JSON structure
    output_json = generate_json_structure(main_directory, truth_txt_path, dataset_num)

    # Output the JSON to a file
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(output_json, file, ensure_ascii=False, indent=4)

    # Print a message indicating completion
    print_l("DEBUG", f"JSON data has been saved to {output_file}")

if __name__ == "__main__":
    main()

