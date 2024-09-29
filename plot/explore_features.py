import json
import glob
import matplotlib
import matplotlib.pyplot as plt
from simplify_factors import simplify_ngram_range, map_feature_type, map_feature_analyzer, map_special_char_and_inc_vocab_richness
# THIS IMPORT WORKS. 
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive plotting

# b1 MIN_VALUE = 1
# b1 PATH = "../../results/baseline-1/experiment_prod_pan20-*.jsonl"

# b2 PATH = "../../results/baseline-2-super/experiment_prod_pan20-super*.jsonl"
# b2 MIN_VALUE = 1

MIN_VALUE=0.86
PATH="../../results/b0-tfidf-experiment-0.jsonl"
data = []

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
    special_char_and_include_vocab_rich = map_special_char_and_inc_vocab_richness(special_char, include_vocab_richness)

    # Check for None values
    if ngram_range is None or feature_analyzer is None or special_char_and_include_vocab_rich is None:
        print(f"Skipping entry due to None values: {entry}")
        print(entry['input_args']['feature_analyzer'])
        print(entry['input_args']['feature_extractor_ngram_range'])
        print(ngram_range)
        continue  # Skip this entry

    x.append(ngram_range)
    y.append(feature_analyzer)
    z.append(special_char_and_include_vocab_rich)
    colors.append(feature_type)

# Create a 3D scatter plot
fig = plt.figure(figsize=(12, 8))  # Increase figure size
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
font_size=20
ax.set_xlabel('Feature Extractor N-gram Range (X)', fontsize=font_size)
ax.set_ylabel('Feature Analyzer (Y)', fontsize=font_size)
ax.set_zlabel('[special_char, include_vocab_richness] (Z)', fontsize=font_size)

# Show plot
plt.subplots_adjust(top=1.2, bottom=-0.2)  
plt.show()