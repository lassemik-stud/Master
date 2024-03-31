import pandas as pd
import json

# Read the JSONL file
with open('evaluation.jsonl', 'r') as file:
    data = [json.loads(line) for line in file]

# Convert the list of dictionaries into a DataFrame
df = pd.json_normalize(data)

# Print the DataFrame

# Rename the columns
df = df.rename(columns={
    'input_args.samples':'Samples',
    'false_negative_rate': 'FNR',
    'true_negative_rate': 'TNR',
    'best_threshold' : 'TH',
    'best_f1_score' : 'F1',
    'auroc': 'AUROC',
    'input_args.clf': 'CLF',
    'input_args.feature_extractor_ngram_range': 'Ngram',
    'input_args.feature_analyzer': 'Analyzer',
    'input_args.special_char': 'SC',
    'input_args.word_length_dist': 'WLD',
    'input_args.include_vocab_richness': 'VR',
    'input_args.svc_c' : 'SVC_C', 
    'input_args.svc_degree' : 'SVC_Degree',
    'input_args.feature_type': 'FT'
})

df = df.sort_values('AUROC', ascending=False)

df = df[df['Samples'] == 1000]
#df = df[df['AUROC'] < 0.8]
#df = df.dropna(subset=['F1','Ngram'])

#df = df['Ngram'].apply(tuple).unique()
#print(df)

# Print the DataFrame with the renamed columns
print(df[['Samples','FNR', 'TNR','TH','F1', 'AUROC', 'CLF', 'SVC_C', 'SVC_Degree', 'Ngram','FT','Analyzer','SC','WLD','VR']])
