import json
import random

from settings.static_values import EXPECTED_PREPROCESSED_DATASETS_FOLDER, EXPECTED_PREPROCESSED_DATASET_FILES
from settings.logging import PrintLog

pan13_train_file = EXPECTED_PREPROCESSED_DATASETS_FOLDER+EXPECTED_PREPROCESSED_DATASET_FILES['pan13-train']
pan13_test_file = EXPECTED_PREPROCESSED_DATASETS_FOLDER+EXPECTED_PREPROCESSED_DATASET_FILES['pan13-test']
pan13_testing_file = EXPECTED_PREPROCESSED_DATASETS_FOLDER+'\\pan13-testing.jsonl'

placeholder = ""

def generate_random_numbers(seed, max_number):
    if max_number < 0:
        PrintLog.error()("Max number must be greater than 0")
        return 0
    try:
        random.seed(seed)
        return random.randint(0, max_number)
    except ValueError as e:
        PrintLog.Info()(f"Error generating random number: {e} - {seed} - {max_number}")

def get_sentences_from_text(text, paragraph_size, unknown_text_location, random_seed=42):
    used_indices = [index for range_ in unknown_text_location for index in range_]
    counter = 0
    while True:
        if counter >= 100:
            PrintLog.error()("Did not find a solution. Adjust the paragraph size or number of sentences", unknown_text_location, random_seed)
            exit()
        start_pos = generate_random_numbers(random_seed, len(text)-paragraph_size-1)
        end_pos = start_pos + paragraph_size
        current_range = list(range(start_pos, end_pos))
        if not any(index in used_indices for index in current_range):
            unknown_text_location.append(current_range)
            return text[start_pos:end_pos], unknown_text_location
        else:
            random_seed += 1
            counter += 1

def add_sentence_to_array(text, paragraph, _known_text_put_location, random_seed=42):
    used_indices = [index for range_ in _known_text_put_location for index in range_]
    counter = 0
    while True:
        if counter >= 50:
            pass
        if counter >= 100:
            PrintLog.error()("Did not find a solution. Adjust the paragraph size or number of sentences", _known_text_put_location)
            exit()
        start_pos = generate_random_numbers(random_seed, len(text)-1)
        end_pos = start_pos + len(paragraph)
        current_range = list(range(start_pos, end_pos))
        if not any(index in used_indices for index in current_range):
            for i, sentence in enumerate(paragraph):
                text.insert(start_pos+i, sentence)
            _known_text_put_location.append(current_range)
            return text, _known_text_put_location
        else:
            random_seed += 1
            counter += 1
    
def test_create_pan13_problem(paragraph_size, number_of_sentences_to_extract,seed=42):
    _problem_text = []
    _location_info = {}
    _unknown_text_retrieval_location = []
    _known_text_put_location = []
    _known_text = [[f"KN{i+1}_S{j+1}" for j in range(200)] for i in range(3)]
    _unknown_text = [f"UN1_S{i+1}" for i in range(200)]
    for i in range(0, number_of_sentences_to_extract):
        _extracted_paragraph,_unknown_text_retrieval_location = get_sentences_from_text(_unknown_text, paragraph_size, _unknown_text_retrieval_location,seed)
        _problem_text,_known_text_put_location = add_sentence_to_array(_known_text[-1],_extracted_paragraph,_known_text_put_location,seed+5555)
        #PrintLog.Info()(f"Extracted paragraph: {_extracted_paragraph}")
    
    PrintLog.debug()(f"Unknown text retrieval location: \t{_unknown_text_retrieval_location}")
    PrintLog.debug()(f"Known text put location: \t\t{_known_text_put_location}")
    PrintLog.debug()("---------------------------------")
    #PrintLog.Info()(f"Problem text: {_problem_text}")
        
        
    
def create_pan13_problem(file_path, paragraph_size=2, number_of_sentences_to_extract=1, max_json_entries=30000, seed=42):
    json_data = []
    with open(file_path, 'r',encoding='utf-8') as file:
        for i, line in enumerate(file):
            if i >= max_json_entries:
                break
            try:
                entry = json.loads(line)
                if len(entry.get('known text')) != 1 and entry.get('same author') == 0:
                    PrintLog.debug()(f"{entry.get('id')}")
                    _unknown_text = entry.get('unknown text')
                    _known_text = entry.get('known text')
                    _problem_text = []
                    _unknown_text_retrieval_location = []
                    _known_text_put_location = []
                    
                    for i in range(0, number_of_sentences_to_extract):
                        _extracted_paragraph,_unknown_text_retrieval_location = get_sentences_from_text(_unknown_text, paragraph_size, _unknown_text_retrieval_location,seed)
                        _problem_text,_known_text_put_location = add_sentence_to_array(_known_text[-1],_extracted_paragraph,_known_text_put_location,seed+5555)
                    
                    PrintLog.debug()(_known_text_put_location)
                    PrintLog.debug()(_unknown_text_retrieval_location)
                    entry = {
                        "id": entry.get('id'),
                        "dataset": entry.get('dataset'),
                        "type": entry.get('type'),  # Assuming first two characters denote the type
                        "author": entry.get('author'),
                        "same author": entry.get('same author'),
                        "known text": _known_text[:-1],
                        "problem_text": _problem_text,
                        "_known_text_put_location" : _known_text_put_location,
                        "_unknown_text_retrieval_location" : _unknown_text_retrieval_location,
                        "additional_info": entry.get('additional_info')
                    }
                    json_data.append(entry)


            except json.JSONDecodeError as e:
                PrintLog.error()(f"Error decoding JSON on line {i+1} - {e}")
            
    with open(EXPECTED_PREPROCESSED_DATASETS_FOLDER+'\\pan13-partial-contract-cheating-test.jsonl', 'w', encoding='utf-8') as file:
        for item in json_data:
            json.dump(item, file, ensure_ascii=False)
            file.write('\n')
    return len(json_data)



   


