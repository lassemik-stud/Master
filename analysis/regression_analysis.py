import json
import glob
import sys
import pandas as pd
from sklearn.preprocessing import MinMaxScaler 
import statsmodels.formula.api as smf
from pygam import LinearGAM

import matplotlib.pyplot as plt  # Import matplotlib

def visualize_relationships(df, title):
    """
    Creates scatter plots of AUC against each predictor variable.

    Args:
        df: The pandas DataFrame containing the data.
        title: The title for the plots (e.g., "Dependency" or "Lexical").
    """

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns
    fig.suptitle(f'AUC vs. Predictor Variables ({title})', fontsize=16)

    for i, predictor in enumerate(['k', 'd', 'x']):
        axes[i].scatter(df[predictor], df['AUC'])
        axes[i].set_xlabel(predictor, fontsize=12)
        axes[i].set_ylabel('AUC', fontsize=12)
        axes[i].set_title(f'AUC vs. {predictor}', fontsize=14)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout for title
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <input_glob>")
        print(f'Example: python3 {sys.argv[0]} "../../results/pan*_b0_*.jsonl"')
        exit()
    else:
        input_glob = sys.argv[1]

    regression_data = []
    for input_file in glob.glob(input_glob):
        with open(input_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                regression_data.append({
                    'k': data['input_args']['ra_k'],
                    'd': data['input_args']['ra_d'],
                    'x': data['input_args']['ra_sentence_size'],
                    'y': data['input_args']['ra_PCC_part_size'],
                    'z' : data['input_args']['ra_number_of_ra_inserts'],
                    'AUC': data['auroc'],
                    'File': input_file 
                })
    
    regression_df = pd.DataFrame(regression_data)
    dep_df = regression_df[regression_df['File'].str.contains('_dep')]
    lex_df = regression_df[regression_df['File'].str.contains('_lex')]
    scaler = MinMaxScaler()
    # regression_df[['k', 'd', 'x']] = scaler.fit_transform(regression_df[['k', 'd', 'x']])
    regression_df[['k', 'd', 'x', 'y', 'z']] = scaler.fit_transform(regression_df[['k', 'd', 'x', 'y', 'z']])

    # Fit the regression model with interaction terms
    # model = smf.ols('AUC ~ k * d * x', data=regression_df).fit()
    model = smf.ols('AUC ~ k * d * x * y * z', data=regression_df).fit()

    # Print the regression results
    print(model.summary())
    #print(regression_df[['k', 'd', 'x', 'AUC']].corr())  # Rolling attribution
    # print(regression_df[['k', 'd', 'x', 'y', 'z', 'AUC']].corr()) # RA with Contract Cheating

    # --- Visualize relationships ---
    # visualize_relationships(dep_df, "Dependency")
    # visualize_relationships(lex_df, "Lexical")

    # print(regression_df[['k', 'd', 'x', 'AUC']].corr())  
    print(regression_df[['k', 'd', 'x', 'y', 'z', 'AUC']].corr())  

    # gam = LinearGAM()
    # gam.fit(regression_df[['k', 'd', 'x']], regression_df['AUC'])
    # gam.summary()
