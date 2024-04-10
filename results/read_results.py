import pandas as pd
import json

# Read the JSONL file
with open('evaluation-2.jsonl', 'r') as file:
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
    'input_args.feature_type': 'FT',
    'time': 'Time',
    'classifier': 'CLF',
    'input_args.feature_extractor_max_features': 'Max_Features',
    'input_args.ra': 'RA',
    'input_args.ra_k': 'RA_K',
    'input_args.ra_d': 'RA_D',
    'input_args.ra_sentence_size': 'RA_Sentence_Size',
    'input_args.clf.SVM': 'SVM',
    'input_args.clf.LR': 'LR',
    'input_args.clf.NaiveBayes': 'NB',
    'input_args.svm_parameters.svm_c': 'SVM_C',
    'input_args.svm_parameters.svm_degree': 'SVM_Degree',
    'input_args.lr_parameters': 'LR_Parameters',
    'input_args.NaiveBayes_parameters': 'NB_Parameters',
    'confusion_matrix.true_negative': 'CM_TN',
    'confusion_matrix.false_positive': 'CM_FP',
    'confusion_matrix.false_negative': 'CM_FN',
    'confusion_matrix.true_positive': 'CM_TP',
    'input_args.ra_PCC_part_size': 'RA_PCC_Part_Size',
    'pcc_results.simple.best_threshold': 'S_BThr',
    'pcc_results.simple.best_f1_score': 'S_Best_F1',
    'pcc_results.simple.false_negative_rate': 'S_FNR',
    'pcc_results.simple.true_negative_rate': 'S_TNR',
    'pcc_results.simple.true_positive_rate': 'S_TPR',
    'pcc_results.simple.false_positive_rate': 'S_FPR',
    'pcc_results.simple.precision': 'S_Precision',
    'pcc_results.simple.recall': 'S_Recall',
    'pcc_results.simple.confusion_matrix.true_negative': 'S_CM_TN',
    'pcc_results.simple.confusion_matrix.false_positive': 'S_CM_FP',
    'pcc_results.simple.confusion_matrix.false_negative': 'S_CM_FN',
    'pcc_results.simple.confusion_matrix.true_positive': 'S_CM_TP',
    'pcc_results.simple.auroc': 'S_AUROC',
    'pcc_results.intermediate.best_threshold': 'Int_BThr',
    'pcc_results.intermediate.best_f1_score': 'Int_Best_F1',
    'pcc_results.intermediate.false_negative_rate': 'Int_FNR',
    'pcc_results.intermediate.true_negative_rate': 'Int_TNR',
    'pcc_results.intermediate.true_positive_rate': 'Int_TPR',
    'pcc_results.intermediate.false_positive_rate': 'Int_FPR',
    'pcc_results.intermediate.precision': 'Int_Precision',
    'pcc_results.intermediate.recall': 'Int_Recall',
    'pcc_results.intermediate.confusion_matrix.true_negative': 'Int_CM_TN',
    'pcc_results.intermediate.confusion_matrix.false_positive': 'Int_CM_FP',
    'pcc_results.intermediate.confusion_matrix.false_negative': 'Int_CM_FN',
    'pcc_results.intermediate.confusion_matrix.true_positive': 'Int_CM_TP',
    'pcc_results.intermediate.auroc': 'Int_AUROC',
    'pcc_results.advanced.correct_location_of_PCC': 'Adv_CL_PCC',
    'pcc_results.advanced.false_location_of_PCC': 'Adv_FL_PCC'
})

df = df.sort_values('AUROC', ascending=False)

#df = df[df['Samples'] == 1000]
#df = df[df['AUROC'] > 0.8]
df = df[df['RA'] == True]
#df = df.dropna(subset=['F1','Ngram'])

#df = df['Ngram'].apply(tuple).unique()
print(df.columns)


# Print the DataFrame with the renamed columns
print(df[['RA', 'RA_PCC_Part_Size', 'RA_K', 'RA_D', 'RA_Sentence_Size', 'Samples','FNR', 'TNR','TH','F1', 'AUROC', 'CLF', 'SVM_C', 'SVM_Degree', 'Ngram','FT','Analyzer','SC','WLD','VR',
          'CM_TN', 'CM_FP', 'CM_FN', 'CM_TP']])


print(df[['S_BThr', 'S_Best_F1', 'S_FNR', 'S_TNR', 'S_TPR', 'S_FPR', 'S_Precision', 'S_Recall', 'S_CM_TN', 'S_CM_FP', 'S_CM_FN', 'S_CM_TP', 'S_AUROC']])

print(df[['Int_BThr', 'Int_Best_F1', 'Int_FNR', 'Int_TNR', 'Int_TPR', 'Int_FPR', 'Int_Precision', 'Int_Recall', 'Int_CM_TN', 'Int_CM_FP', 'Int_CM_FN', 'Int_CM_TP', 'Int_AUROC']])

