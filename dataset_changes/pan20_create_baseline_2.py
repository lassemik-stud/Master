import json
import uuid
import random
import ast
import os

DATASET_PATH = '../../datasets/'
DATASET_CREATE_PATH = 'pan20-test-dataset-1'


def read_jsonl_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line.strip()) for line in file]

def write_jsonl_file(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as file:
        for entry in data:
            file.write(json.dumps(entry) + '\n')

def generate_pairs(same_author_data, different_author_data):   
    same_pairs = []
    different_pairs = []
    i = 0

    for i in range(len(same_author_data)):
        for j in range(i+1, len(same_author_data)):
            if i == j: 
                continue
            same_pairs.append({
                'fandoms': same_author_data[i]['fandoms'],
                'pair': [same_author_data[i]['text'], same_author_data[j]['text']]
            })

    for i in range(len(same_author_data)):
        for j in range(len(different_author_data)):
            if i == j: 
                continue
            different_pairs.append({
                'fandoms': different_author_data[j]['fandoms'],
                'pair': [same_author_data[i]['text'], different_author_data[j]['text']],
                'different_author' : different_author_data[j]['author']
            })

    # while len(same_author_data) > i:
    #     same_pairs.append({
    #         'fandoms': same_author_data[i]['fandoms'],
    #         'pair': [same_author_data[i]['text'], same_author_data[i+1]['text']]
    #     })
    #     i += 2

    #     different_pairs.append({
    #         'fandoms': different_author_data[i]['fandoms'],
    #         'pair': [same_author_data[i]['text'], different_author_data[i]['text']],
    #         'different_author' : different_author_data[i]['author']
    #     })
    #     i += 1
            
    return same_pairs, different_pairs

def main(author_id_list, truth_path, text_path, _type):
    truth_data = read_jsonl_file(truth_path)
    text_data = read_jsonl_file(text_path)

    for author_i, author_id in enumerate(author_id_list):
        
        # Assuming the existence of a function to filter texts by author ID
        same_author_data, different_author_data, all_different_authors = filter_texts_by_author_id(author_id, truth_data, text_data)

        same_pairs, different_pairs = generate_pairs(same_author_data, different_author_data)
        new_text_entries = []
        new_truth_entries = []
        
        # Generate entries for same authorship
        
        for pair in same_pairs:
            entry_id = str(uuid.uuid4())
            fandoms = pair['fandoms']
            pair = pair['pair']
            new_text_entries.append({"id": entry_id, "fandoms": fandoms, "pair": pair})
            new_truth_entries.append({"id": entry_id, "same": True, "authors": [author_id, author_id]})
        
        # Generate entries for different authorship
    
        for pair in different_pairs:
            entry_id = str(uuid.uuid4())
            fandoms = pair['fandoms']
            different_author = pair['different_author']
            pair = pair['pair']
            new_text_entries.append({"id": entry_id, "fandoms": fandoms, "pair": pair})
            new_truth_entries.append({"id": entry_id, "same": False, "authors": [author_id, str(different_author)]})
        
        # Write new JSONL files
        write_jsonl_file(f"{DATASET_PATH}{DATASET_CREATE_PATH}/pan20-{_type}-pairs-{author_id}.jsonl", new_text_entries)
        write_jsonl_file(f"{DATASET_PATH}{DATASET_CREATE_PATH}/pan20-{_type}-truth-{author_id}.jsonl", new_truth_entries)
        path = f'{DATASET_PATH}{DATASET_CREATE_PATH}/pan20-{_type}-all-different-authors.jsonl'
        if not os.path.exists(path):
            write_jsonl_file(path, all_different_authors)
        print(f"{(author_i + 1)} | {len(author_id_list)} \t Finished generating {_type} pairs for author {author_id}")
    
# This function needs to return data with associated fandoms and text pairs for both same and different authors
def filter_texts_by_author_id(author_id, truth_data, text_data):
    same_author = []
    different_author = []

    author_texts_map = {entry["id"]: entry for entry in text_data}
    
    for truth_entry in truth_data:
        if isinstance(truth_entry["authors"], str):
            list_author = [int(i) for i in ast.literal_eval(truth_entry["authors"])]
        elif isinstance(truth_entry["authors"], list):
            list_author = [int(i) for i in truth_entry["authors"]]

        for index, author in enumerate(list_author):
            text_entry = author_texts_map.get(truth_entry["id"])
            author_data = {
                'text': text_entry["pair"][index],
                'fandoms' : tuple(text_entry["fandoms"]),
                'author' : author
            }

            if author == int(author_id):
                same_author.append(author_data)
            else:
                different_author.append(author_data)
           
    # Ensure different_author_data has a variety of authors
    random.shuffle(different_author)
    
    # For simplicity, we might just slice to ensure we don't exceed the number of available texts
    # This assumes there's a substantial mix to potentially have enough for the 10 different author pairs
    different_author_split = different_author[:len(same_author)]
    return same_author, different_author_split, different_author

x_train_path = f"{DATASET_PATH}pan20-authorship-verification-training-small/pan20-authorship-verification-training-small.jsonl"
y_train_path = f"{DATASET_PATH}pan20-authorship-verification-training-small/pan20-authorship-verification-training-small-truth.jsonl"
x_test_path = f"{DATASET_PATH}pan20-authorship-verification-test/pan20-authorship-verification-test.jsonl"
y_test_path = f"{DATASET_PATH}pan20-authorship-verification-test/pan20-authorship-verification-test-truth.jsonl"
os.makedirs(DATASET_PATH+DATASET_CREATE_PATH, exist_ok=True)

if __name__ == "__main__":
    with open('pan20_similar_authors.txt', 'r', encoding='utf-8') as file:
        author_id_list = [line.strip() for line in file if line.strip()]
    
    main(author_id_list=author_id_list, truth_path=y_train_path, text_path=x_train_path, _type='train')
    main(author_id_list=author_id_list, truth_path=y_test_path,  text_path=x_test_path,  _type='test')
