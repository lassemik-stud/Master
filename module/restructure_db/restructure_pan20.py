import json
import sys

from settings.logging import print_l

def read_truth_data(file_path):
    truth_data = {}
    with open(file_path, 'r') as file:
        for line in file:
            record = json.loads(line)
            truth_data[record['id']] = record
    return truth_data

def process_pairs_file(pairs_file_path, truth_data, output_file_path):
    with open(pairs_file_path, 'r') as pairs_file, open(output_file_path, 'w') as output_file:
        for line in pairs_file:
            pair = json.loads(line)
            id = pair['id']

            if id in truth_data:
                combined_entry = {
                    'id': id,
                    'dataset': 20,
                    'type': pair['fandoms'],
                    'author': truth_data[id]['authors'],
                    'same author': 1 if truth_data[id]['same'] else 0,
                    'known text': [pair['pair'][0].split('. ')],
                    'unknown text': pair['pair'][1].split('. ')
                }
                output_file.write(json.dumps(combined_entry) + '\n')

def restructure_pan20_func(path_to_pairs, path_to_truth, output_path):
    truth_data = read_truth_data(path_to_truth)
    process_pairs_file(path_to_pairs, truth_data, output_path)
    print_l("DEBUG", f"JSON data has been saved to {output_path}")

if __name__ == "__main__":
    main()
