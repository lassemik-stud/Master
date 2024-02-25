import uuid
import json
import os
import glob
import re

from settings.logging import print_log

def read_file_content(file_path): 
    with open(os.path.abspath(file_path), 'r', encoding='utf-8') as file:
        return file.read().strip()

def generate_json_structure(directory, truth_file, dataset_number):
    # Load truth data
    with open(os.path.abspath(truth_file), 'r', encoding='utf-8') as file:
        truth_data = file.readlines()
    truth_data = [line.strip() for line in truth_data if line.strip() != ""]
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
                entry["known text"].append(re.split('\. |\.\n',read_file_content(file_path)))

            # Add unknown text
            unknown_file_path = os.path.join(directory, subdir, 'unknown.txt')
            #print(os.path.abspath(unknown_file_path))
            if os.path.exists(os.path.abspath(unknown_file_path)):
                entry["unknown text"] = re.split('\. |\.\n',read_file_content(os.path.abspath(unknown_file_path)))
            else:
                print_log("WARNING", f"Unknown text file not found for {subdir}")

            # Add the entry to the list
            json_data.append(entry)

    return json_data

def restructure_pan13_func(main_directory, truth_txt_path, output_file):
    dataset_num = 13

    # Generate the JSON structure
    output_json = generate_json_structure(main_directory, truth_txt_path, dataset_num)

    # Output the JSON to a file
    with open(output_file, 'w', encoding='utf-8') as file:
        for item in output_json:
            json.dump(item, file, ensure_ascii=False)
            file.write('\n')

    # Print a message indicating completion
    print_log("DEBUG", f"JSON data has been saved to {output_file}")



