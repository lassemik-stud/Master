import json
import matplotlib
import glob
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive plotting

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

MIN_VALUE=1
PATH="../../results/baseline-2-super/experiment_prod_pan20-super-*.jsonl"
data = [] 

# Function to simplify n-gram range
def simplify_ngram_range(ngram_range):
    if ngram_range == [1, 1]:
        return 1
    elif ngram_range == [2, 2]:
        return 2
    elif ngram_range == [3, 3]:
        return 3
    elif ngram_range == [4, 4]:
        return 4
    elif ngram_range == [5, 5]:
        return 5
    elif ngram_range == [1, 2]:
        return 6
    elif ngram_range == [2, 3]:
        return 7
    elif ngram_range == [3, 4]:
        return 8
    elif ngram_range == [4, 5]:
        return 9
    else:
        print(f"Unexpected ngram_range: {ngram_range}")
        return None  # or raise an exception

# Function to map feature type to color
def map_feature_type(feature_type):
    if feature_type == 'tfidf':
        return 'blue'
    elif feature_type == 'BoW':  # Assuming 'BoW' is the other feature type
        return 'red'
    else:
        print(feature_type)
        return 'gray'  # Default color for unexpected types

# Function to map feature analyzer to numerical values
def map_feature_analyzer(feature_analyzer):
    if feature_analyzer == 'word':
        return 2
    elif feature_analyzer == 'char':
        return 4
    elif feature_analyzer == 'char_wb':
        return 6
    return 0 

# Function to calculate extra feature
def calculate_extra(special_char, include_vocab_richness):
    if [special_char,include_vocab_richness] == [True, True]:
        return 6
    elif [special_char,include_vocab_richness] == [True, False]:
        return 2
    elif [special_char,include_vocab_richness] == [False, True]:
        return 4
    elif [special_char,include_vocab_richness] == [False, False]:
        return 0
    else: 
        print(f"unexpected value - [{special_char}, {include_vocab_richness}]")
        return None

# Load data from JSONL file
data = []
# Find all files matching the pattern
files = glob.glob(PATH)

# Iterate over each file
for file_path in files:
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            entry = json.loads(line)
            if entry['auroc'] >= MIN_VALUE:
                data.append(entry)

# Prepare data for plotting
x = []
y = []
z = []
colors = []

for entry in data:
    ngram_range = simplify_ngram_range(entry['input_args']['feature_extractor_ngram_range'])
    feature_type = map_feature_type(entry['input_args']['feature_type'])
    feature_analyzer = map_feature_analyzer(entry['input_args']['feature_analyzer'])
    special_char = entry['input_args']['special_char']
    include_vocab_richness = entry['input_args']['include_vocab_richness']
    extra = calculate_extra(special_char, include_vocab_richness)

    # Check for None values
    if ngram_range is None or feature_analyzer is None or extra is None:
        print(f"Skipping entry due to None values: {entry}")
        print(entry['input_args']['feature_analyzer'])
        print(entry['input_args']['feature_extractor_ngram_range'])
        print(ngram_range)
        continue  # Skip this entry

    x.append(ngram_range)
    y.append(feature_analyzer)
    z.append(extra)
    colors.append(feature_type)

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Use a colormap to map colors
scatter = ax.scatter(x, y, z, c=colors, marker='o')

# Create a legend
legend_labels = ['tfidf', 'BoW']  # Adjust based on your feature types
legend_colors = ['red', 'blue']  # Corresponding colors
for label, color in zip(legend_labels, legend_colors):
    ax.scatter([], [], c=color, label=label)  # Create empty scatter for legend

ax.legend()

# Set labels
ax.set_xlabel('Feature Extractor N-gram Range (X)')
ax.set_ylabel('Feature Analyzer (Y)')
ax.set_zlabel('Extra (Z)')

# Show plot
plt.show()