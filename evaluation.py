import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json 
import datetime

def evaluations(y_test, y_pred_proba, args, classifier_name, PCC_test_params):
    ra = args.get('ra')
    if ra: 
        y_test, y_pred_proba = tranform_ra(PCC_test_params,y_test,y_pred_proba)

    # Calculate precision, recall, and thresholds from the predicted probabilities
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)

    # Calculate F1 scores for each threshold
    epsilon = 1e-10
    f1_scores = 2 * (precision * recall) / (precision + recall + epsilon)

    # Ignore the last value because it is NaN
    f1_scores = f1_scores[:-1]

    # Find the index of the maximum F1 score
    ix = np.argmax(f1_scores)

    # Best threshold and its F1 score
    best_threshold = thresholds[ix]
    best_f1_score = f1_scores[ix]
    best_precision = precision[ix]
    best_recall = recall[ix]

    # Apply the best threshold to predict classes
    y_pred_optimal = (y_pred_proba >= best_threshold).astype(int)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_optimal).ravel()

    # Calculate FNR and TNR
    fnr = fn / (fn + tp)
    tnr = tn / (tn + fp)

    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)

    # Calculate AUROC
    auroc = roc_auc_score(y_test, y_pred_proba)

    # Write to JSON file 
    evaluation_dict = {
    'time' : datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    'classifier': classifier_name,
    'input_args': convert_to_serializable(args),
    'best_threshold': convert_to_serializable(best_threshold),
    'best_f1_score': convert_to_serializable(best_f1_score),
    'false_negative_rate': convert_to_serializable(fnr),
    'true_negative_rate': convert_to_serializable(tnr),
    'true_positive_rate': convert_to_serializable(tpr),
    'false_positive_rate': convert_to_serializable(fpr),
    'precision': convert_to_serializable(best_precision),
    'recall': convert_to_serializable(best_recall),
    'confusion_matrix': {
        'true_negative': convert_to_serializable(tn),
        'false_positive': convert_to_serializable(fp),
        'false_negative': convert_to_serializable(fn),
        'true_positive': convert_to_serializable(tp)
    },
    'auroc': convert_to_serializable(auroc)
}
    # Ensure the 'evaluation' directory exists or adjust the path as needed
    with open('results/evaluation.jsonl', 'a') as f:
        f.write(json.dumps(evaluation_dict) + '\n')

def distribution_plot(y_test, y_pred_proba, arg):

    # Define the figure
    plt.figure(figsize=(10, 6))

    # Plot the distributions
    sns.histplot([prob for prob, actual in zip(y_pred_proba, y_test) if actual == 0], stat='density', kde=True, color='red', label='Different Author')
    sns.histplot([prob for prob, actual in zip(y_pred_proba, y_test) if actual == 1], stat='density', kde=True, color='blue', label='Same Author')

    # Add titles and labels
    plt.title('Predicted Probability Distribution by Actual Class')
    plt.xlabel('Predicted Probability of Same Authorship')
    plt.ylabel('Density')

    # Adding the legend
    plt.legend(title='Actual Class')

    clf = arg.get('clf')
    feature_type = arg.get('feature_type')
    feature_analyzer = arg.get('feature_analyzer')
    feature_extractor_ngram_range = arg.get('feature_extractor_ngram_range')
    ngram_1 = feature_extractor_ngram_range[0] if feature_extractor_ngram_range else 0
    ngram_2 = feature_extractor_ngram_range[1] if feature_extractor_ngram_range else 0
    feature_extractor_max_features = arg.get('feature_extractor_max_features')
    samples = arg.get('samples')
    svc_c = arg.get('svc_c')
    svc_degree = arg.get('svc_degree')
    special_char = arg.get('special_char')
    word_length_dist = arg.get('word_length_dist')
    include_vocab_richness = arg.get('include_vocab_richness')

    # Ensure the 'plot' directory exists or adjust the path as needed
    plt.savefig(f'plot/{samples}{clf}{svc_c}{svc_degree}{feature_type}{feature_analyzer}{ngram_1}{ngram_2}{feature_extractor_max_features}{special_char}{word_length_dist}{include_vocab_richness}class_probability_distribution.pdf', bbox_inches='tight')

    # Close the plot to free memory
    plt.close()


def tranform_ra(PCC_test_params,y_test,y_pred):

    pcc_i = 0
    y_test_pred = []
    y_test_transformed = []

    for pair_i, element in enumerate(PCC_test_params):

        c_size = int(element['l'])
        y_out = y_pred[pcc_i:pcc_i+c_size]
        
        y_truth = y_test[pcc_i:pcc_i+c_size]
        #print(y_truth)
        

        n = element['n']
        k = element['k']
        d = element['d']
        l = element['l']
        N = element['N']
        
        l_group = element['l_group']

        N_theoretical = (n-1)*(k-d) + k
        
        N_val = [[] for _ in range(N_theoretical)]

        count_c = 0
        for elements in range(int(l/n)):
            j = 0
            for element in range(n):
                for i in range(k):
                    N_val[j].append(y_out[count_c])
                    j+=1
                
                count_c+=1
                j-=d
        
        #print(f'RESULTS FOR {pair_i}')
        sum_c = []
        length = 0
        for i, part in enumerate(N_val[:N]):
            array_sum = sum(part)
            length = len(part)
            result = array_sum if array_sum == 0 else array_sum / length
            sum_c.append(result)
            #print(f'--> {round(result,2)} - {y_truth[i]}')
        #print("-----------------------")
        #print(len(y_truth), len(y_test),pcc_i, c_size, N)
        y_test_pred.append(sum(sum_c)/length)
        y_test_transformed.append(y_truth[0])
        
        #print("-----------------------")

        pcc_i+=c_size
    return y_test_transformed, y_test_pred
    
threashold = 0.5
c_value = 0.01
on_change_value = 0.5
cutoff=100
sentence_size = 30
k=5
d=2

def round_by_threashold(value,c):
    global threashold
    if c - on_change_value > value:
        threashold+=c_value
    elif c < value - on_change_value:
        threashold-=c_value
    return 1 if value > threashold else 0

def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.int64):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    else:
        return obj