import json

from settings.structure import Dataset

def read_jsonl(file_path, max_entries):
    entries = []
    with open(file_path, 'r') as file:
        for i, line in enumerate(file):
            if i >= max_entries:
                break
            try:
                entry = json.loads(line)
                entry_id = entry.get('id')
                entry_dataset = entry.get('dataset')
                entry_type = entry.get('type')
                entry_author = entry.get('author')
                entry_classified = entry.get('same author')
                entry_known_text = entry.get('known text')
                entry_unknown_text = entry.get('unknown text')

                known_dict = {}
                unknown_dict = {}
                for text in entry_known_text:
                    known_dict.update({str(text):str(entry_author[0])})
                for text in entry_unknown_text:
                    unknown_dict.update({str(text):str(entry_author[1])})
                
                class_entry = Dataset(entry_id, entry_dataset, entry_type, entry_classified, known_dict, unknown_dict)
                #print(class_entry)
                entries.append(entry)
            except json.JSONDecodeError:
                print(f"Error decoding JSON on line {i+1}")
    return entries

# Usage example:
# entries = read_jsonl('your_file.jsonl', 10)  # Read first 10 entries from the file
