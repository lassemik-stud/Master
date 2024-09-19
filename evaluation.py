import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json 
import datetime
from settings.logging import printLog

import numpy as np
from sklearn.metrics import precision_recall_curve, confusion_matrix, roc_auc_score, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

def evaluation_metrics(y_test, y_pred_proba): 
    if len(set(y_test)) <= 1: 
        best_threshold = "N/A"
        best_f1_score = "N/A"
        fnr = "N/A"
        tnr = "N/A"
        tpr = "N/A"
        fpr = "N/A"
        best_precision = str(1) if y_test == y_pred_proba else str(0)
        best_recall = "N/A"
        tn = "N/A"
        fp = "N/A"
        fn = "N/A"
        tp = "N/A"
        auroc = "N/A"
        return best_threshold, best_f1_score, fnr, tnr, tpr, fpr, best_precision, best_recall, tn, fp, fn, tp, auroc
    
    else:
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

        

        if len(set(y_pred_proba))== 2: 
            y_pred_optimal = y_pred_proba
        else:
            fpr, tpr, threshold_roc = roc_curve(y_test, y_pred_proba)
            fnr = 1 - tpr

            # Find the EER
            eer_threshold = threshold_roc[np.nanargmin(np.abs(fpr - fnr))]
            eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
            y_pred_optimal = (y_pred_proba >= eer_threshold).astype(int)

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_optimal).ravel()

        # Calculate FNR and TNR
        fnr = fn / (fn + tp)
        tnr = tn / (tn + fp)

        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)

        # Calculate AUROC
        auroc = roc_auc_score(y_test, y_pred_proba)

        return best_threshold, best_f1_score, fnr, tnr, tpr, fpr, best_precision, best_recall, tn, fp, fn, tp, auroc

def evaluations(y_test, y_pred_proba, args, classifier_name, PCC_test_params, raw_c_test):
    name = args.get('name')
    ra = args.get('ra')
    if ra: 
        y_test, y_pred_proba, raw_y_test_pred, raw_y = tranform_ra(PCC_test_params,y_test,y_pred_proba, raw_c_test)
    best_threshold, best_f1_score, fnr, tnr, tpr, fpr, best_precision, best_recall, tn, fp, fn, tp, auroc = evaluation_metrics(y_test, y_pred_proba)
    if ra: 
        pcc_simple_test, pcc_simple_pred, pcc_intermediate_pred, pcc_intermediate_test, pcc_advanced = evaluate_pcc(raw_y, raw_y_test_pred, PCC_test_params, best_threshold)
        best_threshold_pcc_simple, best_f1_score_pcc_simple, fnr_pcc_simple, tnr_pcc_simple, tpr_pcc_simple, fpr_pcc_simple, best_precision_pcc_simple, best_recall_pcc_simple, tn_pcc_simple, fp_pcc_simple, fn_pcc_simple, tp_pcc_simple, auroc_pcc_simple = evaluation_metrics(pcc_simple_test, pcc_simple_pred)
        best_threshold_pcc_intermediate, best_f1_score_pcc_intermediate, fnr_pcc_intermediate, tnr_pcc_intermediate, tpr_pcc_intermediate, fpr_pcc_intermediate, best_precision_pcc_intermediate, best_recall_pcc_intermediate, tn_pcc_intermediate, fp_pcc_intermediate, fn_pcc_intermediate, tp_pcc_intermediate, auroc_pcc_intermediate = evaluation_metrics(pcc_intermediate_test, pcc_intermediate_pred)
        correct_location_of_PCC = pcc_advanced.count(1)
        false_location_of_PCC = pcc_advanced.count(0)
        pcc_results = {
            'simple': {
                'best_threshold': convert_to_serializable(best_threshold_pcc_simple),
                'best_f1_score': convert_to_serializable(best_f1_score_pcc_simple),
                'false_negative_rate': convert_to_serializable(fnr_pcc_simple),
                'true_negative_rate': convert_to_serializable(tnr_pcc_simple),
                'true_positive_rate': convert_to_serializable(tpr_pcc_simple),
                'false_positive_rate': convert_to_serializable(fpr_pcc_simple),
                'precision': convert_to_serializable(best_precision_pcc_simple),
                'recall': convert_to_serializable(best_recall_pcc_simple),
                'confusion_matrix': {
                    'true_negative': convert_to_serializable(tn_pcc_simple),
                    'false_positive': convert_to_serializable(fp_pcc_simple),
                    'false_negative': convert_to_serializable(fn_pcc_simple),
                    'true_positive': convert_to_serializable(tp_pcc_simple)
                },
                'auroc': convert_to_serializable(auroc_pcc_simple)
            },
            'intermediate': {
                'best_threshold': convert_to_serializable(best_threshold_pcc_intermediate),
                'best_f1_score': convert_to_serializable(best_f1_score_pcc_intermediate),
                'false_negative_rate': convert_to_serializable(fnr_pcc_intermediate),
                'true_negative_rate': convert_to_serializable(tnr_pcc_intermediate),
                'true_positive_rate': convert_to_serializable(tpr_pcc_intermediate),
                'false_positive_rate': convert_to_serializable(fpr_pcc_intermediate),
                'precision': convert_to_serializable(best_precision_pcc_intermediate),
                'recall': convert_to_serializable(best_recall_pcc_intermediate),
                'confusion_matrix': {
                    'true_negative': convert_to_serializable(tn_pcc_intermediate),
                    'false_positive': convert_to_serializable(fp_pcc_intermediate),
                    'false_negative': convert_to_serializable(fn_pcc_intermediate),
                    'true_positive': convert_to_serializable(tp_pcc_intermediate)
                },
                'auroc': convert_to_serializable(auroc_pcc_intermediate)
            },
            'advanced' : {
                'correct_location_of_PCC': convert_to_serializable(correct_location_of_PCC),
                'false_location_of_PCC': convert_to_serializable(false_location_of_PCC)
            }
        }
    else: 
        pcc_results = "No RA"

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
    'auroc': convert_to_serializable(auroc),
    'pcc_results': pcc_results
    }

    # Ensure the 'evaluation' directory exists or adjust the path as needed
    author_id = args.get('author_id')
    with open(f'results/{str(name)}-{author_id}.jsonl', 'a') as f:
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

def check_for_simple_PCC(array):
    """
    SIMPLE PCC
    evaluate if there is PCC in the set.
    [1,1,1,1] -> 1 -> Not partial contract cheated
    [0,0,0,0] -> 1 -> Not partial contract cheated
    [1,0,0,1] -> 0 -> Partial Contract Cheated
    """
    return 1 if len(set(array)) == 1 else 0

def check_cheating_position(array):
    """
    INTERMEDIATE PCC
    evaluate the location of PCC in the set.
    [1,1,1,0] -> 1 -> RIGHT
    [1,0,1,1] -> 0 -> LEFT
    [1,1,0,0] -> 1 -> RIGHT
    """
    try:
        zero_index = array.index(0)
        return 0 if zero_index < len(array) / 2 else 1
    except ValueError:
        return 1  # No cheating

def check_advanced_PCC(array1, array2):
    return 1 if array1 == array2 else 0

def threashold_func(list, threashold):
    return [1 if x >= threashold else 0 for x in list]

def evaluate_pcc(y_test, y_pred, PCC_test_params, threashold):
    pcc_simple_pred = []
    pcc_simple_test = []

    pcc_intermediate_pred = []
    pcc_intermediate_test = []

    pcc_advanced = []    
    pcc_i = 0
    for pcc_i, pair in enumerate(y_test):
        local_y_pred = threashold_func(y_pred[pcc_i], threashold)
        local_y_test = threashold_func(y_test[pcc_i], threashold)
        simple_pcc_pred = check_for_simple_PCC(local_y_pred)
        simple_pcc_test = check_for_simple_PCC(local_y_test)
        pcc_simple_pred.append(simple_pcc_pred)
        pcc_simple_test.append(simple_pcc_test)

        if simple_pcc_pred == 0 or simple_pcc_test == 0:
            pcc_intermediate_pred.append(local_y_pred)
            pcc_intermediate_test.append(local_y_test)
            pcc_advanced.append(check_advanced_PCC(local_y_pred, local_y_test))

    flat_pcc_intermediate_test = [item for sublist in pcc_intermediate_test for item in (sublist if isinstance(sublist, list) else [sublist])]
    flat_pcc_intermediate_pred = [item for sublist in pcc_intermediate_pred for item in (sublist if isinstance(sublist, list) else [sublist])]


    return pcc_simple_test, pcc_simple_pred, flat_pcc_intermediate_pred, flat_pcc_intermediate_test, pcc_advanced

def evaluate_pcc_intermediate_test(y_test, y_pred, PCC_test_params):
    pcc_intermediate_pred = []
    pcc_intermediate_test = []

    for pair_i, element in enumerate(PCC_test_params):
        length_c = element['N']
        pcc_intermediate_pred.append(sum(y_pred[pair_i:pair_i+length_c])/length_c)
        pcc_intermediate_test.append(sum(y_test[pair_i:pair_i+length_c])/length_c)
        
    return pcc_intermediate_test, pcc_intermediate_pred


def tranform_ra(PCC_test_params,y_test,y_pred, raw_c_test):

    pcc_i = 0
    pcc_raw_counter = 0
    y_test_pred = []
    y_test_transformed = []
    raw_y_test_pred = []
    raw_y = []
    for pair_i, element in enumerate(PCC_test_params):

        c_size = int(element['l'])
        N_size = int(element['N'])
        raw_c_local = raw_c_test[pcc_raw_counter:pcc_raw_counter+N_size]
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
        
        # print(f'RESULTS FOR {pair_i}')
        sum_c = []
        sum_truth = []
        length = 0
        for i, part in enumerate(N_val[:N]):
            array_sum = sum(part)
            length = len(part)
            result = array_sum if array_sum == 0 else array_sum / length
            sum_c.append(result)
            sum_truth.append(raw_c_local[i])
            # print(f'--> {round(result,2)} - {y_truth[i]} - {raw_c_local[i]}')
        # print("-----------------------")
        # print(len(y_truth), len(y_test),pcc_i, c_size, N)
        raw_y_test_pred.append(sum_c)
        raw_y.append(sum_truth)

        y_test_pred.append(sum(sum_c)/length)
        y_test_transformed.append(y_truth[0])

        # print(f'PREDICTED: {sum(sum_c)/length} - {y_truth[0]}')
        # print(f'RAW: {sum_c} - {sum_truth}')
        
        # print("-----------------------")

        pcc_i+=c_size
        pcc_raw_counter+=N_size
    
    flat_raw_y = [item for sublist in raw_y for item in sublist]
    flat_raw_y_test_pred = [item for sublist in raw_y_test_pred for item in sublist]

    y_test_transformed = flat_raw_y
    y_test_pred = flat_raw_y_test_pred

    return y_test_transformed, y_test_pred, raw_y_test_pred, raw_y
    
def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.int64):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    else:
        return obj