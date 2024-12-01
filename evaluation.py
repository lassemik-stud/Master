import json 
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# from scipy.optimize import brentq
# from scipy.interpolate import interp1d

from sklearn.metrics import precision_recall_curve, roc_auc_score, confusion_matrix, roc_curve#,f1_score
#from settings.logging import printLog

from settings.static_values import RESULTS_PATH

def evaluation_metrics(y_test, y_pred_proba, threshold=False): 
    if threshold:
         # Confusion matrix
        if len(set(y_pred_proba)) > 2: 
            y_pred_optimal = (y_pred_proba >= threshold).astype(int) # Apply the threshold to the predicted probabilities
        else: 
            y_pred_optimal = y_pred_proba
        
        #print(y_test, y_pred_optimal)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_optimal).ravel()
        #print(tn, fp, fn, tp)

        # Calculate FNR and TNR
        fnr = fn / (fn + tp)
        tnr = tn / (tn + fp)

        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
    
        return fnr, tnr, tpr, fpr, tn, fp, fn, tp
    
    #print(y_pred_proba)
    #print(len(set(y_test)))
    if len(set(y_test)) <= 1: 
        EER_threshold = "N/A"
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
        fn_zero_threshold = zero_tn = zero_fp = zero_fn = zero_tp = zero_fnr = zero_tnr = zero_tpr = zero_fpr = "N/A"
        return EER_threshold, best_f1_score, fnr, tnr, tpr, fpr, best_precision, best_recall, tn, fp, fn, tp, auroc, zero_tn, zero_fp, zero_fn, zero_tp, zero_fnr, zero_tnr, zero_tpr, zero_fpr, fn_zero_threshold
    
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
        EER_threshold = thresholds[ix]
        best_f1_score = f1_scores[ix]
        best_precision = precision[ix]
        best_recall = recall[ix]

        if len(set(y_pred_proba))== 2: 
            y_pred_optimal = y_pred_proba
            eer_threshold = "N/A"
            fn_zero_threshold = zero_tn = zero_fp = zero_fn = zero_tp = zero_fnr = zero_tnr = zero_tpr = zero_fpr = "N/A"
            #print("y_pred_proba: ", y_pred_proba)
        else:
            fpr, tpr, threshold_roc = roc_curve(y_test, y_pred_proba)
            fnr = 1 - tpr

            # Find the EER
            eer_threshold = threshold_roc[np.nanargmin(np.abs(fpr - fnr))]
            # eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
            y_pred_optimal = (y_pred_proba >= eer_threshold).astype(int)

            # Initialize variables to store the highest threshold and confusion matrix
            fn_zero_threshold = None
            conf_matrix = None
            for threshold in sorted(threshold_roc, reverse=True):
                # Calculate predicted labels
                y_pred_optimal_FN = (y_pred_proba >= threshold).astype(int)
                
                # Compute confusion matrix
                tn, fp, fn, tp = confusion_matrix(y_test, y_pred_optimal_FN).ravel()
                
                # Check if FN is zero
                if fn == 0:
                    fn_zero_threshold = threshold
                    zero_tn = tn
                    zero_fp = fp
                    zero_fn = fn
                    zero_tp = tp
                    zero_fnr = zero_fn / (zero_fn + zero_tp)
                    zero_tnr = zero_tn / (zero_tn + zero_fp)
                    zero_tpr = zero_tp / (zero_tp + zero_fn)
                    zero_fpr = zero_fp / (zero_fp + zero_tn)
                    break
                else:
                    fn_zero_threshold = zero_tn = zero_fp = zero_fn = zero_tp = zero_fnr = zero_tnr = zero_tpr = zero_fpr = "N/A"

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_optimal).ravel()

        # Calculate FNR and TNR
        fnr = fn / (fn + tp)
        tnr = tn / (tn + fp)

        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        
        # Calculate AUROC
        auroc = roc_auc_score(y_test, y_pred_proba)
        return eer_threshold, best_f1_score, fnr, tnr, tpr, fpr, best_precision, best_recall, tn, fp, fn, tp, auroc, zero_tn, zero_fp, zero_fn, zero_tp, zero_fnr, zero_tnr, zero_tpr, zero_fpr, fn_zero_threshold

def evaluations(y_test, y_pred_proba, args, classifier_name, pcc_test_params, raw_c_test):
    distribution_plot_v = args.get('distribution_plot')
    
    name = args.get('name')
    ra = args.get('ra')
    if ra: 
        y_test, y_pred_proba, raw_y_test_pred, raw_y = tranform_ra(pcc_test_params,y_test,y_pred_proba, raw_c_test)
    EER_threshold, best_f1_score, fnr, tnr, tpr, fpr, best_precision, best_recall, tn, fp, fn, tp, auroc,zero_tn, zero_fp, zero_fn, zero_tp, zero_fnr, zero_tnr, zero_tpr, zero_fpr, fn_zero_threshold = evaluation_metrics(y_test, y_pred_proba)

    if distribution_plot_v:
        if bool(distribution_plot_v):
            distribution_plot(y_test, y_pred_proba, args, EER_threshold, fn_zero_threshold)
    
    if ra: 
        pcc_simple_test, pcc_simple_pred, pcc_intermediate_pred, pcc_intermediate_test, pcc_advanced = evaluate_pcc(raw_y, raw_y_test_pred, pcc_test_params, EER_threshold)
  
        #print("SIMPLE EER")
        fnr_pcc_simple, tnr_pcc_simple, tpr_pcc_simple, fpr_pcc_simple, tn_pcc_simple, fp_pcc_simple, fn_pcc_simple, tp_pcc_simple = evaluation_metrics(pcc_simple_test, pcc_simple_pred, EER_threshold)
        #print("INTERMEDIATE EER")
        fnr_pcc_intermediate, tnr_pcc_intermediate, tpr_pcc_intermediate, fpr_pcc_intermediate, tn_pcc_intermediate, fp_pcc_intermediate, fn_pcc_intermediate, tp_pcc_intermediate = evaluation_metrics(pcc_intermediate_test, pcc_intermediate_pred, EER_threshold)

        zero_FP_pcc_simple_test, zero_FP_simple_pred, zero_FP_intermediate_pred, zero_FP_intermediate_test, zero_FP_advanced = evaluate_pcc(raw_y, raw_y_test_pred, pcc_test_params, fn_zero_threshold)
        
        #print("SIMPLE FN ZERO")
        zerofp_fnr_pcc_simple, zerofp_tnr_pcc_simple, zerofp_tpr_pcc_simple, zerofp_fpr_pcc_simple, zerofp_tn_pcc_simple, zerofp_fp_pcc_simple, zerofp_fn_pcc_simple, zerofp_tp_pcc_simple = evaluation_metrics(zero_FP_pcc_simple_test, zero_FP_simple_pred, fn_zero_threshold)
        #print("INTERMEDIATE FN ZERO")
        zerofp_fnr_pcc_intermediate, zerofp_tnr_pcc_intermediate, zerofp_tpr_pcc_intermediate, zerofp_fpr_pcc_intermediate, zerofp_tn_pcc_intermediate, zerofp_fp_pcc_intermediate, zerofp_fn_pcc_intermediate, zerofp_tp_pcc_intermediate = evaluation_metrics(zero_FP_intermediate_test, zero_FP_intermediate_pred, fn_zero_threshold)
        
        pcc_results = {
            'simple_EER': {
                'EER_threshold': convert_to_serializable(EER_threshold),
                'false_negative_rate': convert_to_serializable(fnr_pcc_simple),
                'true_negative_rate': convert_to_serializable(tnr_pcc_simple),
                'true_positive_rate': convert_to_serializable(tpr_pcc_simple),
                'false_positive_rate': convert_to_serializable(fpr_pcc_simple),
                'confusion_matrix': {
                    'true_negative': convert_to_serializable(tn_pcc_simple),
                    'false_positive': convert_to_serializable(fp_pcc_simple),
                    'false_negative': convert_to_serializable(fn_pcc_simple),
                    'true_positive': convert_to_serializable(tp_pcc_simple)
                },
                'simple_FN_zero_threshold': {
                    'false_negative_rate': convert_to_serializable(zerofp_fnr_pcc_simple),
                    'true_negative_rate': convert_to_serializable(zerofp_tnr_pcc_simple),
                    'true_positive_rate': convert_to_serializable(zerofp_tpr_pcc_simple),
                    'false_positive_rate': convert_to_serializable(zerofp_fpr_pcc_simple),
                    'confusion_matrix': {
                        'true_negative': convert_to_serializable(zerofp_tn_pcc_simple),
                        'false_positive': convert_to_serializable(zerofp_fp_pcc_simple),
                        'false_negative': convert_to_serializable(zerofp_fn_pcc_simple),
                        'true_positive': convert_to_serializable(zerofp_tp_pcc_simple)
                },
                }
            },
            'intermediate_EER': {
                'EER_threshold': convert_to_serializable(EER_threshold),
                'false_negative_rate': convert_to_serializable(fnr_pcc_intermediate),
                'true_negative_rate': convert_to_serializable(tnr_pcc_intermediate),
                'true_positive_rate': convert_to_serializable(tpr_pcc_intermediate),
                'false_positive_rate': convert_to_serializable(fpr_pcc_intermediate),
                'confusion_matrix': {
                    'true_negative': convert_to_serializable(tn_pcc_intermediate),
                    'false_positive': convert_to_serializable(fp_pcc_intermediate),
                    'false_negative': convert_to_serializable(fn_pcc_intermediate),
                    'true_positive': convert_to_serializable(tp_pcc_intermediate)
                },
                'intermediate_FN_zero_threshold': {
                    'false_negative_rate': convert_to_serializable(zerofp_fnr_pcc_intermediate),
                    'true_negative_rate': convert_to_serializable(zerofp_tnr_pcc_intermediate),
                    'true_positive_rate': convert_to_serializable(zerofp_tpr_pcc_intermediate),
                    'false_positive_rate': convert_to_serializable(zerofp_fpr_pcc_intermediate),
                    'confusion_matrix': {
                        'true_negative': convert_to_serializable(zerofp_tn_pcc_intermediate),
                        'false_positive': convert_to_serializable(zerofp_fp_pcc_intermediate),
                        'false_negative': convert_to_serializable(zerofp_fn_pcc_intermediate),
                        'true_positive': convert_to_serializable(zerofp_tp_pcc_intermediate)
                },
                    'threshold': convert_to_serializable(fn_zero_threshold)
                
                    
                }
            }
        }
        #print(pcc_results)
        
    else: 
        pcc_results = "No RA"

    # Write to JSON file 
    evaluation_dict = {
    'time' : datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    'classifier': classifier_name,
    'input_args': convert_to_serializable(args),
    'EER_threshold': convert_to_serializable(EER_threshold),
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
    'FN_zero_threshold': {
        'true_negative': convert_to_serializable(zero_tn),
        'false_positive': convert_to_serializable(zero_fp),
        'false_negative': convert_to_serializable(zero_fn),
        'true_positive': convert_to_serializable(zero_tp),
        'false_negative_rate': convert_to_serializable(zero_fnr),
        'true_negative_rate': convert_to_serializable(zero_tnr),
        'true_positive_rate': convert_to_serializable(zero_tpr),
        'false_positive_rate': convert_to_serializable(zero_fpr),
        'threshold': convert_to_serializable(fn_zero_threshold)
    },
    'pcc_results': pcc_results
    }

    # Ensure the 'evaluation' directory exists or adjust the path as needed
    author_id = args.get('author_id')
    with open(f'{RESULTS_PATH}{str(name)}-{author_id}.jsonl', 'a', encoding='utf-8') as f:
        f.write(json.dumps(evaluation_dict) + '\n')

def get_best_auroc(args, current_auroc):
    args = args[0]
    name = args.get('name')
    author_id = args.get('author_id')
    
    try:
        with open(f'{RESULTS_PATH}{str(name)}-{author_id}.jsonl', 'r', encoding='utf-8') as f:
            last_200_lines = []
            for line in f:
                line = line.strip()
                if line:  # Check if the line is not empty
                    try:
                        evaluation = json.loads(line)
                        last_200_lines.append(evaluation)
                        if len(last_200_lines) > 200:
                            last_200_lines.pop(0)  # Keep only the last 200
                    except json.JSONDecodeError:
                        print(f"Skipping invalid JSON: {line}") 

            for evaluation in last_200_lines:
                if evaluation['auroc'] > current_auroc:
                    current_auroc = evaluation['auroc']
                    best_classifier = evaluation['classifier']  # Store the entire evaluation
                else:
                    best_classifier = ""
        
        return current_auroc, best_classifier
    
    except FileNotFoundError:
        print(f"File not found: {RESULTS_PATH}{str(name)}-{author_id}.jsonl")
        return None

def distribution_plot(y_test, y_pred_proba, args, EER_threshold, FN_zero_threshold):
    name = args.get('name')
    author_id = args.get('author_id')
    # Define the figure
    plt.figure(figsize=(10, 6))

    # Plot the distributions
    sns.histplot([prob for prob, actual in zip(y_pred_proba, y_test) if actual == 0], stat='density', kde=True, color='red', label='Different Author')
    sns.histplot([prob for prob, actual in zip(y_pred_proba, y_test) if actual == 1], stat='density', kde=True, color='blue', label='Same Author')
    
    # Add the vertical line for the best threshold
    plt.axvline(x=EER_threshold, color='black', linestyle='--', label=f'EER Threshold: {EER_threshold:.3f}') 
    plt.axvline(x=FN_zero_threshold, color='red', linestyle='--', label=f'FP=0 Threshold: {FN_zero_threshold:.3f}') 
    
    # Add titles and labels
    plt.title('Predicted Probability Distribution by Actual Class')
    plt.xlabel('Predicted Probability of Same Authorship')
    plt.ylabel('Number of samples')

    # Adding the legend
    plt.legend(title='')

    # Ensure the 'plot' directory exists or adjust the path as needed
    plt.savefig(f'{RESULTS_PATH}/plot/{str(name)}-{author_id}-class_probability_distribution.pdf', bbox_inches='tight')

    # Close the plot to free memory
    plt.close()

def check_for_simple_pcc(array):
    """
    SIMPLE PCC
    evaluate if there is PCC in the set.
    [1,1,1,1] -> 1 -> Not partial contract cheated
    [0,0,0,0] -> 1 -> Not partial contract cheated
    [1,0,0,1] -> 0 -> Partial Contract Cheated
    """
    return 1 if len(set(array)) == 1 else 0

 ######################### NOT USED #########################
def check_cheating_position(array): # NOT USED
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
######################### NOT USED #########################

def check_advanced_pcc(array1, array2):
    return 1 if array1 == array2 else 0

def threashold_func(list, threashold):
    return [1 if x >= threashold else 0 for x in list]

def evaluate_pcc(y_test, y_pred, pcc_test_params, threashold):
    #y_test)
    pcc_simple_pred = []
    pcc_simple_test = []

    pcc_intermediate_pred = []
    pcc_intermediate_test = []

    pcc_advanced = []    
    pcc_i = 0
    for pcc_i, pair in enumerate(y_test):
        local_y_pred = threashold_func(y_pred[pcc_i], threashold)
        local_y_test = threashold_func(y_test[pcc_i], threashold)
        simple_pcc_pred = check_for_simple_pcc(local_y_pred)
        simple_pcc_test = check_for_simple_pcc(local_y_test)
        pcc_simple_pred.append(simple_pcc_pred)
        pcc_simple_test.append(simple_pcc_test)

        if simple_pcc_pred == 0 or simple_pcc_test == 0:
            pcc_intermediate_pred.append(local_y_pred)
            pcc_intermediate_test.append(local_y_test)
            pcc_advanced.append(check_advanced_pcc(local_y_pred, local_y_test))

    flat_pcc_intermediate_test = [item for sublist in pcc_intermediate_test for item in (sublist if isinstance(sublist, list) else [sublist])]
    flat_pcc_intermediate_pred = [item for sublist in pcc_intermediate_pred for item in (sublist if isinstance(sublist, list) else [sublist])]

    return pcc_simple_test, pcc_simple_pred, flat_pcc_intermediate_pred, flat_pcc_intermediate_test, pcc_advanced

def evaluate_pcc_intermediate_test(y_test, y_pred, pcc_test_params):
    pcc_intermediate_pred = []
    pcc_intermediate_test = []

    for pair_i, element in enumerate(pcc_test_params):
        length_c = element['N']
        pcc_intermediate_pred.append(sum(y_pred[pair_i:pair_i+length_c])/length_c)
        pcc_intermediate_test.append(sum(y_test[pair_i:pair_i+length_c])/length_c)
        
    return pcc_intermediate_test, pcc_intermediate_pred


def tranform_ra(pcc_test_params,y_test,y_pred, raw_c_test):

    pcc_i = 0
    pcc_raw_counter = 0
    y_test_pred = []
    y_test_transformed = []
    raw_y_test_pred = []
    raw_y = []
    for pair_i, element in enumerate(pcc_test_params):

        c_size = int(element['l'])
        n_size = int(element['N'])
        raw_c_local = raw_c_test[pcc_raw_counter:pcc_raw_counter+n_size]
        y_out = y_pred[pcc_i:pcc_i+c_size]
        
        y_truth = y_test[pcc_i:pcc_i+c_size]
        #print(y_truth)
        
        n = element['n']
        k = element['k']
        d = element['d']
        l = element['l']
        N = element['N']
        
        l_group = element['l_group']

        n_theoretical = (n-1)*(k-d) + k
        
        N_val = [[] for _ in range(n_theoretical)]

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
            #print(f'--> {round(result,2)} - {y_truth[i]} - {raw_c_local[i]}')
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
        pcc_raw_counter+=n_size
    
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