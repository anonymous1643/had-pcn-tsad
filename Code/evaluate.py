
from __future__ import annotations

import copy
import json
import numpy as np
import pandas as pd
from sklearn import metrics

from had_pcn_import_style import get_artifacts


def pak(preds, targets, k=20):
    predicts = np.asarray(preds).astype(int).copy()
    actuals = np.asarray(targets).astype(int)
    one_start_idx = np.where(np.diff(actuals, prepend=0) == 1)[0]
    zero_start_idx = np.where(np.diff(actuals, prepend=0) == -1)[0]
    assert len(one_start_idx) == len(zero_start_idx) + 1 or len(one_start_idx) == len(zero_start_idx)
    if len(one_start_idx) == len(zero_start_idx) + 1:
        zero_start_idx = np.append(zero_start_idx, len(predicts))
    for i in range(len(one_start_idx)):
        start = one_start_idx[i]
        end = zero_start_idx[i]
        seg_len = end - start
        if seg_len > 0 and predicts[start:end].sum() > (k / 100.0) * seg_len:
            predicts[start:end] = 1
    return predicts


def get_events(y_test, outlier=1, normal=0):
    y_test = np.asarray(y_test).astype(int)
    events = {}
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
    if len(y_test) > 0 and label_prev == outlier:
        event_end = len(y_test) - 1
        events[event] = (event_start, event_end)
    return events


def get_composite_fscore_raw(pred_labels, true_events, y_test, return_prec_rec=False):
    epsilon = 1e-8
    pred_labels = np.asarray(pred_labels).astype(int)
    y_test = np.asarray(y_test).astype(int)
    tp = np.sum([pred_labels[start:end + 1].any() for start, end in true_events.values()])
    fn = len(true_events) - tp
    rec_e = tp / (tp + fn + epsilon)
    prec_t = metrics.precision_score(y_test, pred_labels, zero_division=0)
    fscore_c = 2 * rec_e * prec_t / (rec_e + prec_t + epsilon)
    if prec_t == 0 and rec_e == 0:
        fscore_c = 0.0
    if return_prec_rec:
        return prec_t, rec_e, fscore_c
    return fscore_c


def constant_bias_fn(inputs):
    inputs = np.asarray(inputs)
    return 0.0 if inputs.shape[0] == 0 else np.sum(inputs) / inputs.shape[0]


def improved_cardinality_fn(cardinality, gt_length):
    return 0.0 if gt_length <= 0 else ((gt_length - 1) / gt_length) ** (cardinality - 1)


def compute_window_indices(binary_labels):
    binary_labels = np.asarray(binary_labels).astype(int)
    boundaries = np.empty_like(binary_labels)
    boundaries[0] = 0
    boundaries[1:] = binary_labels[:-1]
    boundaries *= -1
    boundaries += binary_labels
    indices = np.nonzero(boundaries)[0].tolist()
    if len(indices) % 2 != 0:
        indices.append(binary_labels.shape[0])
    return [(indices[i], indices[i + 1]) for i in range(0, len(indices), 2)]


def _compute_overlap(preds, pred_indices, gt_indices, alpha, bias_fn, cardinality_fn, use_window_weight=False):
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
    denom = total_gt_points if use_window_weight else max(n_gt_windows, 1)
    return total_score / denom


def ts_precision_and_recall(anomalies, predictions, alpha=0, recall_bias_fn=constant_bias_fn, recall_cardinality_fn=improved_cardinality_fn, precision_bias_fn=None, precision_cardinality_fn=None, anomaly_ranges=None, prediction_ranges=None, weighted_precision=False):
    anomalies = np.asarray(anomalies).astype(int)
    predictions = np.asarray(predictions).astype(int)
    has_anomalies = np.any(anomalies > 0)
    has_predictions = np.any(predictions > 0)
    if not has_predictions and not has_anomalies:
        return 1.0, 1.0
    if not has_predictions or not has_anomalies:
        return 0.0, 0.0
    if precision_bias_fn is None:
        precision_bias_fn = recall_bias_fn
    if precision_cardinality_fn is None:
        precision_cardinality_fn = recall_cardinality_fn
    if anomaly_ranges is None:
        anomaly_ranges = compute_window_indices(anomalies)
    if prediction_ranges is None:
        prediction_ranges = compute_window_indices(predictions)
    recall = _compute_overlap(predictions, prediction_ranges, anomaly_ranges, alpha, recall_bias_fn, recall_cardinality_fn)
    precision = _compute_overlap(anomalies, anomaly_ranges, prediction_ranges, 0, precision_bias_fn, precision_cardinality_fn, use_window_weight=weighted_precision)
    return precision, recall


class AnoEvaluator:
    def __init__(self, preds, targets):
        self.preds = np.asarray(preds).astype(int)
        self.targets = np.asarray(targets).astype(int)
        assert len(self.preds) == len(self.targets)

    def eval_naive_f1(self):
        f1 = metrics.f1_score(self.targets, self.preds, zero_division=0)
        prec = metrics.precision_score(self.targets, self.preds, zero_division=0)
        rec = metrics.recall_score(self.targets, self.preds, zero_division=0)
        return f1, prec, rec

    def eval_pak_auc(self):
        pak_metrics_list = []
        ks = np.arange(0, 110, 10)
        for k in ks:
            preds_new = copy.deepcopy(self.preds)
            targets_new = copy.deepcopy(self.targets)
            pa_scores = pak(preds_new, targets_new, k)
            pak_metrics_list.append([
                metrics.f1_score(targets_new, pa_scores, zero_division=0),
                metrics.precision_score(targets_new, pa_scores, zero_division=0),
                metrics.recall_score(targets_new, pa_scores, zero_division=0),
            ])
        pak_metrics = np.array(pak_metrics_list)
        f1_pak_auc = metrics.auc(ks, pak_metrics[:, 0]) / 100.0
        prec_pak_auc = metrics.auc(ks, pak_metrics[:, 1]) / 100.0
        rec_pak_auc = metrics.auc(ks, pak_metrics[:, 2]) / 100.0
        return f1_pak_auc, prec_pak_auc, rec_pak_auc

    def eval_f1_composite(self):
        true_events = get_events(self.targets)
        prec_pw, rec_ew, f1_comp = get_composite_fscore_raw(self.preds, true_events, self.targets, return_prec_rec=True)
        return f1_comp, prec_pw, rec_ew

    def eval_f1_range(self):
        epsilon = 1e-8
        label_ranges = compute_window_indices(self.targets)
        prec, rec = ts_precision_and_recall(self.targets, self.preds, alpha=0, anomaly_ranges=label_ranges, weighted_precision=True)
        f1 = 2 * prec * rec / (prec + rec + epsilon)
        return f1, prec, rec

    def full_results(self):
        result_f1, prec, rec = self.eval_naive_f1()
        result_f1_pak, pre_pak, rec_pak = self.eval_pak_auc()
        result_f1_comp, pre_comp, rec_comp = self.eval_f1_composite()
        result_f1_range, pre_range, rec_range = self.eval_f1_range()
        return {
            "F1@K": round(result_f1_pak, 5),
            "F1-C": round(result_f1_comp, 5),
            "F1-R": round(result_f1_range, 5),
            "F1": round(result_f1, 5),
            "Prec": round(prec, 5),
            "Rec": round(rec, 5),
            "Prec@K": round(pre_pak, 5),
            "Rec@K": round(rec_pak, 5),
            "Prec-C": round(pre_comp, 5),
            "Rec-C": round(rec_comp, 5),
            "Prec-R": round(pre_range, 5),
            "Rec-R": round(rec_range, 5),
        }


def continual_learning_metrics(task_matrix):
    a = np.asarray(task_matrix, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("task_matrix must be square")
    t = a.shape[0]
    final_f1 = float(np.mean(a[-1, :]))
    aa = float(np.mean(a[np.tril_indices(t)]))
    forgetting_vals = []
    bwt_vals = []
    fwt_vals = []
    for j in range(t):
        final_perf = a[-1, j]
        initial_perf = a[j, j]
        if j < t - 1:
            best_after_learn = np.max(a[j:, j])
            forgetting_vals.append(best_after_learn - final_perf)
            bwt_vals.append(final_perf - initial_perf)
        if j > 0:
            fwt_vals.append(a[j - 1, j])
    forgetting = float(np.mean(forgetting_vals)) if forgetting_vals else 0.0
    bwt = float(np.mean(bwt_vals)) if bwt_vals else 0.0
    fwt = float(np.mean(fwt_vals)) if fwt_vals else 0.0
    return {"Final F1": round(final_f1, 5), "AA": round(aa, 5), "F": round(forgetting, 5), "BWT": round(bwt, 5), "FWT": round(fwt, 5)}


def main():
    artifacts = get_artifacts(force_retrain=False, verbose=True)
    y_test = artifacts["y_test"]
    rows = []

    evaluator_r = AnoEvaluator(artifacts["reactive_test_preds"], y_test)
    out_r = evaluator_r.full_results()
    out_r["Method"] = "Ours (R)"
    out_r["Threshold"] = round(float(artifacts["reactive_threshold"]), 6)
    rows.append(out_r)

    evaluator_p = AnoEvaluator(artifacts["proactive_test_preds"], y_test)
    out_p = evaluator_p.full_results()
    out_p["Method"] = "Ours (P, H=25)"
    out_p["Threshold"] = round(float(artifacts["proactive_threshold"]), 6)
    rows.append(out_p)

    paper_df = pd.DataFrame(rows)[["Method", "F1@K", "F1-C", "F1-R", "F1", "Prec", "Rec", "Prec@K", "Rec@K", "Prec-C", "Rec-C", "Prec-R", "Rec-R", "Threshold"]]
    print("\nAnomaly Detection Metricsn")
    print(paper_df.to_string(index=False))

    cl_rows = []
    reactive_cl = continual_learning_metrics(artifacts["reactive_task_matrix"])
    reactive_cl["Mode"] = "Reactive"
    cl_rows.append(reactive_cl)

    proactive_cl = continual_learning_metrics(artifacts["proactive_task_matrix"])
    proactive_cl["Mode"] = "Proactive"
    cl_rows.append(proactive_cl)

    cl_df = pd.DataFrame(cl_rows)[["Mode", "Final F1", "AA", "F", "BWT", "FWT"]]
    print("\nContinual Learning Metricsn")
    print(cl_df.to_string(index=False))




if __name__ == "__main__":
    main()
