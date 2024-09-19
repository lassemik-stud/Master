import json
import uuid
import random
import ast
import os

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
        write_jsonl_file(f"../datasets/pan20-created-test/pan20-{_type}-pairs-{author_id}.jsonl", new_text_entries)
        write_jsonl_file(f"../datasets/pan20-created-test/pan20-{_type}-truth-{author_id}.jsonl", new_truth_entries)
        path = f'../datasets/pan20-created-test/pan20-{_type}-all-different-authors.jsonl'
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

root = '/home/lasse'
x_train_path = f"{root}/datasets/pan20-authorship-verification-training-small/pan20-authorship-verification-training-small.jsonl"
y_train_path = f"{root}/datasets/pan20-authorship-verification-training-small/pan20-authorship-verification-training-small-truth.jsonl"
x_test_path = f"{root}/datasets/pan20-authorship-verification-test/pan20-authorship-verification-test.jsonl"
y_test_path = f"{root}/datasets/pan20-authorship-verification-test/pan20-authorship-verification-test-truth.jsonl"

if __name__ == "__main__":
    author_id_list = [
    "1000555",
    "1004266",
    "1044982",
    "1046661",
    "1059049",
    "1060901",
    "1067919",
    "1102473",
    "1124370",
    "1134135",
    "1144748",
    "1207016",
    "1236246",
    "1254171",
    "1259527",
    "134389",
    "1354544",
    "1384142",
    "139401",
    "1431943",
    "146806",
    "1492084",
    "150067",
    "1555294",
    "1576308",
    "159540",
    "1597786",
    "1641001",
    "1648312",
    "1652711",
    "1655407",
    "1716956",
    "1777261",
    "1783027",
    "1796268",
    "1810674",
    "182830",
    "1862439",
    "1869762",
    "187436",
    "1952016",
    "1956189",
    "1957052",
    "2002255",
    "2007348",
    "2031530",
    "204149",
    "2049660",
    "2129042",
    "2135508",
    "214030",
    "2213298",
    "2299844",
    "2352342",
    "2376938",
    "240162",
    "2456602",
    "2469390",
    "25619",
    "2664779",
    "2669603",
    "267735",
    "2688002",
    "270548",
    "2738227",
    "2762778",
    "283936",
    "284145",
    "296467",
    "298653",
    "3090681",
    "3107154",
    "31351",
    "32276",
    "3231678",
    "324872",
    "3561385",
    "357927",
    "3628045",
    "3667168",
    "3669238",
    "3735343",
    "3993743",
    "404703",
    "429953",
    "4339208",
    "4373288",
    "437416",
    "4415171",
    "442738",
    "44720",
    "4483094",
    "4787616",
    "480321",
    "4865253",
    "526713",
    "5430304",
    "547570",
    "55318",
    "561615",
    "56264",
    "578300",
    "607817",
    "610733",
    "627559",
    "646233",
    "649516",
    "70311",
    "709114",
    "744563",
    "74824",
    "763713",
    "76380",
    "80018",
    "806976",
    "870118",
    "882056",
    "900596",
    "909661",
    "913162",
    "9154517",
    "920809",
    "951853",
    "974478"
]
    truth_path = y_train_path
    text_path = x_train_path
    main(author_id_list, truth_path, text_path, 'train')

    truth_path = y_test_path
    text_path = x_test_path
    main(author_id_list, truth_path, text_path, 'test')
