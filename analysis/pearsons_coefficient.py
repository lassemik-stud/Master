import json
import glob
import sys
from scipy.stats import pearsonr
import pandas as pd
import numpy as np

def calculate_pearson_coefficients(filename):
    """
    Calculates Pearson correlation coefficients between k, d, x, and AUC 
    from data in a JSONL file.

    Args:
    filename: The path to the JSONL file.

    Returns:
    A dictionary containing the Pearson coefficients for each variable.
    """

    k_values = []
    d_values = []
    x_values = []
    auc_values = []

    with open(filename, 'r') as f:
        for line in f:
            data = json.loads(line)
            k_values.append(data['input_args']['ra_k'])
            d_values.append(data['input_args']['ra_d'])
            x_values.append(data['input_args']['ra_sentence_size'])
            auc_values.append(data['auroc'])
            
    correlations = {
        'k': pearsonr(k_values, auc_values),
        'd': pearsonr(d_values, auc_values),
        'x': pearsonr(x_values, auc_values),
    }
    auc_mean = np.mean(auc_values)

    return correlations, auc_mean

if __name__ == '__main__':
    print(len(sys.argv))
    if len(sys.argv) != 2:  # Corrected the number of arguments to 3
        print("Usage: python script_name.py <input_glob>")
        print(f'Example: python3 {sys.argv[0]} "../../results/pan*_b0_*.jsonl"') 
        exit()
    else:
        input_glob = sys.argv[1]
    
    all_data = []  # List to store data for the DataFrame

    for input_file in glob.glob(input_glob):
        print(f"Processing file: {input_file}")
        correlations, auc_mean = calculate_pearson_coefficients(input_file)
        for var, (coeff, p_value) in correlations.items():
            #print(f"\tPearson coefficient between {var} and AUC: {coeff:.3f} (p-value: {p_value:.3f})")
            all_data.append([input_file, var, coeff, p_value, auc_mean])
    
    df = pd.DataFrame(all_data, columns=['File', 'Variable', 'Coefficient', 'P-value', 'AUC-Mean'])
    df = df[df['P-value'] < 0.05]
    df = df[df['AUC-Mean'] > 0.8]
    df.sort_values(by='Variable', inplace=True, ascending=False)
    print(df.to_markdown(index=False, numalign="left", stralign="left"))

