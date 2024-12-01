import json
import sys

def dataset_statistics(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def count_words(text):
    words = text.split()
    return len(words)

def count_characters(text):
    return len(text)

def count_stats(data):
    word_count = 0
    char_count = 0
    discourse_counts = {}
    total_texts = 0  # Keep track of the total number of texts
    for entry in data:
        for text in entry['pair']:
            word_count += count_words(text)
            char_count += count_characters(text)
            total_texts += 1  # Increment for each text in the pair
        for discourse_type in entry['fandoms']:
            discourse_counts[discourse_type] = discourse_counts.get(discourse_type, 0) + 1

    # Calculate averages
    avg_words_per_text = word_count / total_texts if total_texts else 0
    avg_chars_per_text = char_count / total_texts if total_texts else 0

    return word_count, char_count, discourse_counts, avg_words_per_text, avg_chars_per_text

def run(file_path):
    data = dataset_statistics(file_path)
    
    word_count, char_count, discourse_counts, avg_words, avg_chars = count_stats(data)

    print(f"Number of samples: {len(data)}")
    print(f"Number of words: {word_count}")
    print(f"Number of characters: {char_count}")
    print(f"Discourse type counts: {discourse_counts}")
    print(f"Average words per text: {avg_words:.2f}")  # Format to 2 decimal places
    print(f"Average characters per text: {avg_chars:.2f}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <file path>")
    else:
        file_path = sys.argv[1]
        run(file_path)