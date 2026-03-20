# # This code is from  the research paper:
# "Possibility for Proactive Anomaly Detection"
# Code published at OpenReview: https://openreview.net/forum?id=w63aCqNRFp
# The scoring metrics for this code were used for consistency and fairness.

import numpy as np
import os
from pathlib import Path
from matplotlib import pyplot
import pandas as pd
from sklearn.mixture import GaussianMixture
from pyod.models.ecod import ECOD
from joblib import dump, load
import copy
from pyod.models.deep_svdd import DeepSVDD

from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
# util functions about data

from scipy.stats import rankdata, iqr, trim_mean
from sklearn.metrics import f1_score, mean_squared_error
import numpy as np
from numpy import percentile


def get_attack_interval(attack): 
    heads = []
    tails = []
    for i in range(len(attack)):
        if attack[i] == 1:
            if attack[i-1] == 0:
                heads.append(i)
            
            if i < len(attack)-1 and attack[i+1] == 0:
                tails.append(i)
            elif i == len(attack)-1:
                tails.append(i)
    res = []
    for i in range(len(heads)):
        res.append((heads[i], tails[i]))
    # print(heads, tails)
    return res

# calculate F1 scores
def eval_scores(scores, true_scores, th_steps, return_thresold=False):
    padding_list = [0]*(len(true_scores) - len(scores))
    # print(padding_list)

    if len(padding_list) > 0:
        scores = padding_list + scores

    scores_sorted = rankdata(scores, method='ordinal')
    th_steps = th_steps
    # th_steps = 500
    th_vals = np.array(range(th_steps)) * 1.0 / th_steps
    fmeas = [None] * th_steps
    thresholds = [None] * th_steps
    for i in range(th_steps):
        cur_pred = scores_sorted > th_vals[i] * len(scores)

        fmeas[i] = f1_score(true_scores, cur_pred)

        score_index = scores_sorted.tolist().index(int(th_vals[i] * len(scores)+1))
        thresholds[i] = scores[score_index]

    if return_thresold:
        return fmeas, thresholds
    return fmeas

def eval_scores2(scores, true_scores, th_steps, return_thresold=False):
    padding_list = [0]*(len(true_scores) - len(scores))

    if len(padding_list) > 0:
        scores = padding_list + scores

    scores_sorted = rankdata(scores, method='ordinal')
    th_steps = th_steps
    # th_steps = 500
    th_vals = np.array(range(th_steps)) * 1.0 / th_steps
    fmeas = [None] * th_steps
    thresholds = [None] * th_steps
    for i in range(th_steps):
        cur_pred = scores_sorted < th_vals[i] * len(scores)

        fmeas[i] = f1_score(true_scores, cur_pred)

        score_index = scores_sorted.tolist().index(int(th_vals[i] * len(scores)+1))
        thresholds[i] = scores[score_index]

    if return_thresold:
        return fmeas, thresholds
    return fmeas

def eval_mseloss(predicted, ground_truth):

    ground_truth_list = np.array(ground_truth)
    predicted_list = np.array(predicted)

    
    # mask = (ground_truth_list == 0) | (predicted_list == 0)

    # ground_truth_list = ground_truth_list[~mask]
    # predicted_list = predicted_list[~mask]

    # neg_mask = predicted_list < 0
    # predicted_list[neg_mask] = 0

    # err = np.abs(predicted_list / ground_truth_list - 1)
    # acc = (1 - np.mean(err))

    # return loss
    loss = mean_squared_error(predicted_list, ground_truth_list)

    return loss

def get_err_median_and_iqr(predicted, groundtruth):

    np_arr = np.abs(np.subtract(np.array(predicted), np.array(groundtruth)))

    err_median = np.median(np_arr)
    err_iqr = iqr(np_arr)

    return err_median, err_iqr

def get_err_median_and_iqr_gt(predicted):

    # np_arr = np.abs(np.array(predicted))
    np_arr = np.array(predicted)

    err_median = np.median(np_arr)
    err_iqr = iqr(np_arr)

    return err_median, err_iqr

def get_err_median_and_quantile(predicted, groundtruth, percentage):

    np_arr = np.abs(np.subtract(np.array(predicted), np.array(groundtruth)))

    err_median = np.median(np_arr)
    # err_iqr = iqr(np_arr)
    err_delta = percentile(np_arr, int(percentage*100)) - percentile(np_arr, int((1-percentage)*100))

    return err_median, err_delta

def get_err_mean_and_quantile(predicted, groundtruth, percentage):

    np_arr = np.abs(np.subtract(np.array(predicted), np.array(groundtruth)))

    err_median = trim_mean(np_arr, percentage)
    # err_iqr = iqr(np_arr)
    err_delta = percentile(np_arr, int(percentage*100)) - percentile(np_arr, int((1-percentage)*100))

    return err_median, err_delta

def get_err_mean_and_std(predicted, groundtruth):

    np_arr = np.abs(np.subtract(np.array(predicted), np.array(groundtruth)))

    err_mean = np.mean(np_arr)
    err_std = np.std(np_arr)

    return err_mean, err_std

def get_err_mean_and_std_gt(predicted):

    # np_arr = np.abs(np.array(predicted))
    np_arr = np.array(predicted)

    err_mean = np.mean(np_arr)
    err_std = np.std(np_arr)

    return err_mean, err_std


def get_f1_score(scores, gt, contamination):

    padding_list = [0]*(len(gt) - len(scores))
    # print(padding_list)

    threshold = percentile(scores, 100 * (1 - contamination))

    if len(padding_list) > 0:
        scores = padding_list + scores

    pred_labels = (scores > threshold).astype('int').ravel()

    return f1_score(gt, pred_labels)


##### Evaluation Metrics #####

def pak(preds, targets, k=20):

    predicts = preds
    actuals = targets

    one_start_idx = np.where(np.diff(actuals, prepend=0) == 1)[0]
    zero_start_idx = np.where(np.diff(actuals, prepend=0) == -1)[0]

    assert len(one_start_idx) == len(zero_start_idx) + 1 or len(one_start_idx) == len(zero_start_idx)

    if len(one_start_idx) == len(zero_start_idx) + 1:
        zero_start_idx = np.append(zero_start_idx, len(predicts))

    for i in range(len(one_start_idx)):
        if predicts[one_start_idx[i]:zero_start_idx[i]].sum() > k / 100 * (zero_start_idx[i] - one_start_idx[i]):
            predicts[one_start_idx[i]:zero_start_idx[i]] = 1

    return predicts

def get_events(y_test, outlier=1, normal=0):
    events = dict()
    label_prev = normal
    event = 0 
    event_start = 0
    for tim, label in enumerate(y_test):
        if label == outlier:
            if label_prev == normal:
                event += 1
                event_start = tim
        else:
            if label_prev == outlier:
                event_end = tim - 1
                events[event] = (event_start, event_end)
        label_prev = label

    if label_prev == outlier:
        event_end = tim - 1
        events[event] = (event_start, event_end)
    return events

def get_composite_fscore_raw(pred_labels, true_events, y_test, return_prec_rec=False):
    epsilon = 1e-8
    tp = np.sum([pred_labels[start:end + 1].any() for start, end in true_events.values()])
    fn = len(true_events) - tp
    rec_e = tp/(tp + fn)
    prec_t = metrics.precision_score(y_test, pred_labels, zero_division=0)
    fscore_c = 2 * rec_e * prec_t / (rec_e + prec_t + epsilon)
    if prec_t == 0 and rec_e == 0:
        fscore_c = 0
    if return_prec_rec:
        return prec_t, rec_e, fscore_c
    return fscore_c

def constant_bias_fn(inputs):

    return np.sum(inputs) / inputs.shape[0]

def improved_cardinality_fn(cardinality, gt_length):
    
    return ((gt_length - 1) / gt_length) ** (cardinality - 1)

def compute_window_indices(binary_labels):
    
    boundaries = np.empty_like(binary_labels)
    boundaries[0] = 0
    boundaries[1:] = binary_labels[:-1]
    boundaries *= -1
    boundaries += binary_labels

    indices = np.nonzero(boundaries)[0].tolist()
    if len(indices) % 2 != 0:
        indices.append(binary_labels.shape[0])
    indices = [(indices[i], indices[i + 1]) for i in range(0, len(indices), 2)]

    return indices

def _compute_overlap(preds, pred_indices, gt_indices, alpha, bias_fn, cardinality_fn, use_window_weight = False):
    n_gt_windows = len(gt_indices)
    n_pred_windows = len(pred_indices)
    total_score = 0.0
    total_gt_points = 0

    i = j = 0
    while i < n_gt_windows and j < n_pred_windows:
        gt_start, gt_end = gt_indices[i]
        window_length = gt_end - gt_start
        total_gt_points += window_length
        i += 1

        cardinality = 0
        while j < n_pred_windows and pred_indices[j][1] <= gt_start:
            j += 1
        while j < n_pred_windows and pred_indices[j][0] < gt_end:
            j += 1
            cardinality += 1

        if cardinality == 0:
            continue
        
        j -= 1

        cardinality_multiplier = cardinality_fn(cardinality, window_length)

        prediction_inside_ground_truth = preds[gt_start:gt_end]
        omega = bias_fn(prediction_inside_ground_truth)

        weight = window_length if use_window_weight else 1

        total_score += alpha * weight
        total_score += (1 - alpha) * cardinality_multiplier * omega * weight

    denom = total_gt_points if use_window_weight else n_gt_windows

    return total_score / denom

def ts_precision_and_recall(anomalies, predictions, alpha = 0,
                            recall_bias_fn = constant_bias_fn,
                            recall_cardinality_fn = improved_cardinality_fn,
                            precision_bias_fn = None,
                            precision_cardinality_fn = None,
                            anomaly_ranges = None,
                            prediction_ranges = None,
                            weighted_precision = False):
    has_anomalies = np.any(anomalies > 0)
    has_predictions = np.any(predictions > 0)

    if not has_predictions and not has_anomalies:
        return 1, 1
    elif not has_predictions or not has_anomalies:
        return 0, 0

    if precision_bias_fn is None:
        precision_bias_fn = recall_bias_fn
    if precision_cardinality_fn is None:
        precision_cardinality_fn = recall_cardinality_fn

    if anomaly_ranges is None:
        anomaly_ranges = compute_window_indices(anomalies)
    if prediction_ranges is None:
        prediction_ranges = compute_window_indices(predictions)

    recall = _compute_overlap(predictions, prediction_ranges, anomaly_ranges, alpha, recall_bias_fn,
                              recall_cardinality_fn)
    precision = _compute_overlap(anomalies, anomaly_ranges, prediction_ranges, 0, precision_bias_fn,
                                 precision_cardinality_fn, use_window_weight=weighted_precision)

    return precision, recall

##### Evaluation function #####

class ano_evaluator:
    def __init__(self, preds, targets = None):
        assert len(preds) == len(targets)
        
        try:
            preds = np.asarray(preds)
            targets = np.asarray(targets)
        except TypeError:
            preds = np.asarray(preds.cpu())
            targets = np.asarray(targets.cpu())
            
        self.targets = targets
        self.preds = preds
        
    def eval_naive_f1(self):
        f1 = metrics.f1_score(self.targets, self.preds, zero_division = 0)
        prec = metrics.precision_score(self.targets, self.preds, zero_division=0)
        recall = metrics.recall_score(self.targets, self.preds, zero_division=0)
        return f1, prec, recall
        
    def eval_pak_auc(self):
        pak_metrics_list = []
        for k in np.arange(0,110, 10):
            preds_new = copy.deepcopy(self.preds)
            targets_new = copy.deepcopy(self.targets)
            pa_scores = pak(preds_new, targets_new, k)
            pak_metrics_list.append([metrics.f1_score(targets_new, pa_scores, zero_division = 0),
                                     metrics.precision_score(targets_new, pa_scores, zero_division=0),
                                     metrics.recall_score(targets_new, pa_scores, zero_division=0),
                                     ])
        pak_metrics = np.array(pak_metrics_list)
        f1_pak_auc = metrics.auc(np.arange(0,110, 10), pak_metrics[:,0]) / 100.0
        prec_pak_auc = metrics.auc(np.arange(0,110, 10), pak_metrics[:,1]) / 100.0
        recall_pak_auc = metrics.auc(np.arange(0,110, 10), pak_metrics[:,2]) / 100.0

        return f1_pak_auc, prec_pak_auc, recall_pak_auc
    
    def eval_f1_composite(self):
        true_events = get_events(self.targets)
        prec_pw, rec_ew, f1_comp = get_composite_fscore_raw(self.preds, true_events, self.targets, return_prec_rec=True)
        return f1_comp, prec_pw, rec_ew
    
    def eval_f1_range(self):
        epsilon = 1e-8
        label_ranges = compute_window_indices(self.targets)
        prec, recall = ts_precision_and_recall(self.targets, self.preds, alpha=0,
                                                anomaly_ranges=label_ranges,
                                                weighted_precision=True)
        f1 = (1 + 1**2) * prec * recall / (1**2 * prec + recall + epsilon) 
        return f1, prec, recall
    


##### Output function #####

def return_scores(evaluator):
    result_f1, prec, rec = evaluator.eval_naive_f1()
    result_f1, prec, rec = round(result_f1, 5), round(prec, 5),round(rec, 5)
    
    result_f1_pak, pre_pak, rec_pak = evaluator.eval_pak_auc()
    result_f1_pak, pre_pak, rec_pak = round(result_f1_pak, 5), round(pre_pak, 5),round(rec_pak, 5)
    
    result_f1_comp, pre_comp, rec_comp = evaluator.eval_f1_composite()
    result_f1_comp, pre_comp, rec_comp = round(result_f1_comp, 5), round(pre_comp, 5),round(rec_comp, 5)
    
    result_f1_range, pre_range, rec_range = evaluator.eval_f1_range()
    result_f1_range, pre_range, rec_range = round(result_f1_range, 5), round(pre_range, 5),round(rec_range, 5)
    
    return result_f1, prec, rec, result_f1_pak, pre_pak, rec_pak, result_f1_comp, pre_comp, rec_comp, result_f1_range, pre_range, rec_range

def norm(train, test):

    normalizer = MinMaxScaler(feature_range=(0, 1)).fit(train) # scale training data to [0,1] range
    train_ret = normalizer.transform(train)
    test_ret = normalizer.transform(test)

    return train_ret, test_ret


##### Main evaluation function #####

def get_score_2(test_pred_init, test_labels, train_orig, dataset, save_folder):
    norm_train, norm_test  = norm(train_orig, test_pred_init)
    result_list = []
    
        ######################  GMM = norm ######################
    feature_num = norm_train.shape[1]
    if os.path.isfile(f'./pretrained/Unsupervised_model/{dataset}/gmm.joblib'):
        gmm = load(f'./pretrained/Unsupervised_model/{dataset}/gmm.joblib')
        print(f"Get pretrained GMM model")
    else:
        print(f"No pretrained GMM model : conduct GMM model fitting")
        gmm = GaussianMixture(n_components=feature_num, n_init=5, random_state=42)
        gmm= gmm.fit(norm_train) 
        dirname = os.path.dirname(f'./pretrained/Unsupervised_model/{dataset}/gmm.joblib')
        Path(dirname).mkdir(parents=True, exist_ok=True)
        dump(gmm, f'./pretrained/Unsupervised_model/{dataset}/gmm.joblib')
    
    X_score_gmm = gmm.score_samples(norm_train)
    Y_score_gmm = gmm.score_samples(norm_test)
    threshold_gmm = np.min(X_score_gmm)
    Y_label_gmm = Y_score_gmm < threshold_gmm
    Y_label_gmm = np.where(Y_label_gmm==True,1,0)
    
    evaluator_gmm = ano_evaluator(Y_label_gmm, test_labels)
    result_f1_gmm, pre_gmm, rec_gmm, result_f1_pak_gmm, pre_pak_gmm, rec_pak_gmm, result_f1_comp_gmm, pre_comp_gmm, rec_comp_gmm,result_f1_range_gmm, pre_range_gmm, rec_range_gmm = return_scores(evaluator_gmm)
    
    result_list.append(["gmm", result_f1_gmm, pre_gmm, rec_gmm, result_f1_pak_gmm, pre_pak_gmm, rec_pak_gmm
                        ,result_f1_comp_gmm, pre_comp_gmm, rec_comp_gmm, result_f1_range_gmm, pre_range_gmm, rec_range_gmm])

    ######################  DeepSVDD ######################
    # Check if pretrained DeepSVDD model exists
    model_path = f'./pretrained/Unsupervised_model/{dataset}/DeepSVDD.joblib'
    if os.path.isfile(model_path):
        DeepSVDD_model = load(model_path)
        print("Loaded pretrained DeepSVDD model")
    else:
        print("No pretrained DeepSVDD model: fitting DeepSVDD model")
        DeepSVDD_model = DeepSVDD(
            n_features=norm_train.shape[1],  # dynamically determine number of features
            contamination=0.05,
            epochs=10,
            verbose=1
        )
        DeepSVDD_model.fit(norm_train)
        # Ensure directory exists
        dirname = os.path.dirname(model_path)
        Path(dirname).mkdir(parents=True, exist_ok=True)
        dump(DeepSVDD_model, model_path)
    
    # Compute anomaly scores
    X_score_DeepSVDD = DeepSVDD_model.decision_function(norm_train)
    Y_score_DeepSVDD = DeepSVDD_model.decision_function(norm_test)

    # Threshold and prediction
    threshold_DeepSVDD = np.max(X_score_DeepSVDD)
    Y_label_DeepSVDD = DeepSVDD_model.predict(norm_test)
    
    # Evaluate predictions
    evaluator_DeepSVDD = ano_evaluator(Y_label_DeepSVDD, test_labels)
    (
        result_f1_DeepSVDD, pre_DeepSVDD, rec_DeepSVDD,
        result_f1_pak_DeepSVDD, pre_pak_DeepSVDD, rec_pak_DeepSVDD,
        result_f1_comp_DeepSVDD, pre_comp_DeepSVDD, rec_comp_DeepSVDD,
        result_f1_range_DeepSVDD, pre_range_DeepSVDD, rec_range_DeepSVDD
    ) = return_scores(evaluator_DeepSVDD)
    
    # Append results
    result_list.append([
        "DeepSVDD",
        result_f1_DeepSVDD, pre_DeepSVDD, rec_DeepSVDD,
        result_f1_pak_DeepSVDD, pre_pak_DeepSVDD, rec_pak_DeepSVDD,
        result_f1_comp_DeepSVDD, pre_comp_DeepSVDD, rec_comp_DeepSVDD,
        result_f1_range_DeepSVDD, pre_range_DeepSVDD, rec_range_DeepSVDD
    ])
    ######################  ECOD = norm ######################
    if os.path.isfile(f'./pretrained/Unsupervised_model/{dataset}/ECOD.joblib'):
        ECOD_model = load(f'./pretrained/Unsupervised_model/{dataset}/ECOD.joblib')
        print("Get pretrained ECOD model")
    else:
        print("No pretrained ECOD model : conduct ECOD model fitting")
        ECOD_model = ECOD(contamination=0.001)
        ECOD_model= ECOD_model.fit(norm_train)
        dirname = os.path.dirname(f'./pretrained/Unsupervised_model/{dataset}/ECOD.joblib')
        Path(dirname).mkdir(parents=True, exist_ok=True)
        dump(ECOD_model, f'./pretrained/Unsupervised_model/{dataset}/ECOD.joblib')
        
    X_score_ECOD = ECOD_model.decision_function(norm_train) ; Y_score_ECOD = ECOD_model.decision_function(norm_test)
    threshold_ECOD = np.max(X_score_ECOD)  ;  Y_label_ECOD = ECOD_model.predict(norm_test)
    
    evaluator_ECOD = ano_evaluator(Y_label_ECOD, test_labels)
    result_f1_ECOD, pre_ECOD, rec_ECOD, result_f1_pak_ECOD, pre_pak_ECOD, rec_pak_ECOD, result_f1_comp_ECOD, pre_comp_ECOD, rec_comp_ECOD ,result_f1_range_ECOD, pre_range_ECOD, rec_range_ECOD  = return_scores(evaluator_ECOD)
    
    result_list.append(["ECOD", result_f1_ECOD, pre_ECOD, rec_ECOD, result_f1_pak_ECOD, pre_pak_ECOD, rec_pak_ECOD, result_f1_comp_ECOD, pre_comp_ECOD, rec_comp_ECOD ,
                        result_f1_range_ECOD, pre_range_ECOD, rec_range_ECOD])
    
    return result_list
