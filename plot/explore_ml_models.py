import json
import glob
import matplotlib
import matplotlib.pyplot as plt
from simplify_factors import simplify_ngram_range, map_feature_type, map_feature_analyzer, map_special_char_and_inc_vocab_richness
from simplify_factors import simplify_svm_kernel, simplify_svm_c, simplify_ngram_range
from simplify_factors import simplify_lr_c, simplify_nb_alpha, simplify_nb_fit_prior
# THIS IMPORT WORKS.

matplotlib.use('TkAgg')  # Use TkAgg backend for interactive plotting

# b1 MIN_VALUE = 1
# b1 PATH = "../../results/baseline-1/experiment_prod_pan20-*.jsonl"

# b2 PATH = "../../results/baseline-2-super/experiment_prod_pan20-super*.jsonl"
# b2 MIN_VALUE = 1

MIN_VALUE=0.87
PATH="../../results/b0-tfidf-experiment-0.jsonl"
CLASSIFIER = "SVM" # SVM or NaiveBayes
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
            if entry['auroc'] >= MIN_VALUE and entry['classifier'] == CLASSIFIER:
                data.append(entry)

# Prepare data for plotting
x = []
y = []
z = []
colors = []

for entry in data:
    if CLASSIFIER == "SVM":
        svm_c = simplify_svm_c(entry['input_args']['svm_parameters']['svm_c'])
        svm_degree = entry['input_args']['svm_parameters']['svm_degree']
        svm_kernel = simplify_svm_kernel(entry['input_args']['svm_parameters']['svm_kernel'])

        x.append(svm_c)
        y.append(svm_degree)
        z.append(svm_kernel)

    elif CLASSIFIER == "LR":
        lr_c = simplify_lr_c(entry['input_args']['lr_parameters']['lr_c'])
        lr_l1_ratio = entry['input_args']['lr_parameters']['lr_l1_ratio']

        x.append(lr_c)
        y.append(lr_l1_ratio)

    elif CLASSIFIER == "NaiveBayes":
        nb_alpha = simplify_nb_alpha(entry['input_args']['naiveBayes_parameters']['nb_alpha'])
        nb_fit_prior = simplify_nb_fit_prior(entry['input_args']['naiveBayes_parameters']['nb_fit_prior'])

        x.append(nb_alpha)
        y.append(nb_fit_prior)
    else:
        print(f"Unexpected classifier: {CLASSIFIER}")
        continue


if CLASSIFIER == "SVM":
    # Create a 3D scatter plot
    fig = plt.figure(figsize=(12, 8))  # Increase figure size
    ax = fig.add_subplot(111, projection='3d')

    # Use a colormap to map colors
    scatter = ax.scatter(x, y, z, marker='o')

    ax.legend()

    # Set labels
    font_size=20
    ax.set_xlabel('SVM C', fontsize=font_size)
    ax.set_ylabel('SVM Degree', fontsize=font_size)
    ax.set_zlabel('SVM Kernel', fontsize=font_size)

    # Show plot
    plt.subplots_adjust(top=1.2, bottom=-0.2)  
    plt.show()

elif CLASSIFIER == "LR":
    # Create a 3D scatter plot
    fig = plt.figure(figsize=(12, 8))  # Increase figure size
    ax = fig.add_subplot(111, projection='rectilinear') 

    # Use a colormap to map colors
    scatter = ax.scatter(x, y, marker='o')

    ax.legend()

    # Set labels
    font_size=20
    ax.set_xlabel('LR C', fontsize=font_size)
    ax.set_ylabel('LR Ratio', fontsize=font_size)

    # Show plot
    plt.subplots_adjust(top=1.2, bottom=-0.2)  
    plt.show()
elif CLASSIFIER == "NaiveBayes":
    # Create a 3D scatter plot
    fig = plt.figure(figsize=(12, 8))  # Increase figure size
    ax = fig.add_subplot(111, projection='rectilinear') 

    # Use a colormap to map colors
    scatter = ax.scatter(x, y, marker='o')

    ax.legend()

    # Set labels
    font_size=20
    ax.set_xlabel('NB alpha', fontsize=font_size)
    ax.set_ylabel('NB Fit prior', fontsize=font_size)

    # Show plot
    plt.subplots_adjust(top=1.2, bottom=-0.2)  
    plt.show()