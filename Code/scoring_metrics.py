# # This code is from  the research paper:
# "Possibility for Proactive Anomaly Detection"
# Code published at OpenReview: https://openreview.net/forum?id=w63aCqNRFp
# The scoring metrics for this code were used for consistency and fairness.

import copy
import numpy as np
import pandas as pd
from sklearn import metrics



# Basic helpers
def get_attack_interval(attack):
    """
    Convert binary labels into a list of inclusive anomaly intervals.
    Example: [0,1,1,0,1] -> [(1,2), (4,4)]
    """
    attack = np.asarray(attack).astype(int)
    heads = []
    tails = []

    for i in range(len(attack)):
        if attack[i] == 1:
            if i == 0 or attack[i - 1] == 0:
                heads.append(i)

            if i < len(attack) - 1 and attack[i + 1] == 0:
                tails.append(i)
            elif i == len(attack) - 1:
                tails.append(i)

    return list(zip(heads, tails))


def scores_to_ranks(scores):
    """
    Convert scores to ordinal ranks.
    Higher score => larger rank.
    """
    scores = np.asarray(scores).ravel()
    order = scores.argsort(kind="mergesort")
    ranks = np.empty_like(order, dtype=int)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


def threshold_from_train_scores(train_scores, method="max_normal", contamination=None):
    """
    Choose a threshold from train scores.

    Convention:
      larger score = more anomalous
    """
    train_scores = np.asarray(train_scores).ravel()

    if method == "max_normal":
        return float(np.max(train_scores))

    if method == "percentile":
        if contamination is None:
            raise ValueError("contamination must be provided for percentile thresholding")
        return float(np.percentile(train_scores, 100 * (1 - contamination)))

    raise ValueError(f"Unknown threshold method: {method}")


def scores_to_labels(scores, threshold, higher_is_more_anomalous=True):
    """
    Turn anomaly scores into binary labels.
    """
    scores = np.asarray(scores).ravel()
    if higher_is_more_anomalous:
        return (scores > threshold).astype(int)
    return (scores < threshold).astype(int)


# F1 sweep helpers 

def eval_scores(scores, true_scores, th_steps=500, return_threshold=False):
    """
    Sweep thresholds over ranked scores and compute point-wise F1.
    Higher score = more anomalous.
    """
    scores = list(np.asarray(scores).ravel())
    true_scores = np.asarray(true_scores).astype(int).ravel()

    padding_list = [0] * (len(true_scores) - len(scores))
    if len(padding_list) > 0:
        scores = padding_list + scores

    scores_sorted = scores_to_ranks(scores)
    th_vals = np.arange(th_steps) / float(th_steps)

    fmeas = [None] * th_steps
    thresholds = [None] * th_steps

    for i in range(th_steps):
        cur_pred = (scores_sorted > th_vals[i] * len(scores)).astype(int)
        fmeas[i] = metrics.f1_score(true_scores, cur_pred, zero_division=0)

        target_rank = int(th_vals[i] * len(scores) + 1)
        score_index = np.where(scores_sorted == target_rank)[0][0]
        thresholds[i] = scores[score_index]

    if return_threshold:
        return fmeas, thresholds
    return fmeas


def eval_scores2(scores, true_scores, th_steps=500, return_threshold=False):
    """
    Sweep thresholds over ranked scores and compute point-wise F1.
    Lower score = more anomalous.
    """
    scores = list(np.asarray(scores).ravel())
    true_scores = np.asarray(true_scores).astype(int).ravel()

    padding_list = [0] * (len(true_scores) - len(scores))
    if len(padding_list) > 0:
        scores = padding_list + scores

    scores_sorted = scores_to_ranks(scores)
    th_vals = np.arange(th_steps) / float(th_steps)

    fmeas = [None] * th_steps
    thresholds = [None] * th_steps

    for i in range(th_steps):
        cur_pred = (scores_sorted < th_vals[i] * len(scores)).astype(int)
        fmeas[i] = metrics.f1_score(true_scores, cur_pred, zero_division=0)

        target_rank = int(th_vals[i] * len(scores) + 1)
        score_index = np.where(scores_sorted == target_rank)[0][0]
        thresholds[i] = scores[score_index]

    if return_threshold:
        return fmeas, thresholds
    return fmeas


# PaK / event-adjusted metric

def pak(preds, targets, k=20):
    """
    Point-adjusted-k postprocessing.
    If predicted positives inside a ground-truth anomaly range exceed k%,
    mark the whole range as positive.
    """
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
        if seg_len <= 0:
            continue

        if predicts[start:end].sum() > (k / 100.0) * seg_len:
            predicts[start:end] = 1

    return predicts


def get_events(y_test, outlier=1, normal=0):
    """
    Convert binary labels into event dict:
      {1: (start, end), 2: (start, end), ...}
    with inclusive endpoints.
    """
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
    """
    Composite F1:
      - precision at point level
      - recall at event level
    """
    pred_labels = np.asarray(pred_labels).astype(int)
    y_test = np.asarray(y_test).astype(int)

    epsilon = 1e-8
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


# Range-based metric

def constant_bias_fn(inputs):
    inputs = np.asarray(inputs)
    if inputs.shape[0] == 0:
        return 0.0
    return np.sum(inputs) / inputs.shape[0]


def improved_cardinality_fn(cardinality, gt_length):
    if gt_length <= 0:
        return 0.0
    return ((gt_length - 1) / gt_length) ** (cardinality - 1)


def compute_window_indices(binary_labels):
    """
    Convert binary labels into half-open windows:
      [start, end)
    """
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


def _compute_overlap(
    preds,
    pred_indices,
    gt_indices,
    alpha,
    bias_fn,
    cardinality_fn,
    use_window_weight=False,
):
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


def ts_precision_and_recall(
    anomalies,
    predictions,
    alpha=0,
    recall_bias_fn=constant_bias_fn,
    recall_cardinality_fn=improved_cardinality_fn,
    precision_bias_fn=None,
    precision_cardinality_fn=None,
    anomaly_ranges=None,
    prediction_ranges=None,
    weighted_precision=False,
):
    anomalies = np.asarray(anomalies).astype(int)
    predictions = np.asarray(predictions).astype(int)

    has_anomalies = np.any(anomalies > 0)
    has_predictions = np.any(predictions > 0)

    if not has_predictions and not has_anomalies:
        return 1.0, 1.0
    elif not has_predictions or not has_anomalies:
        return 0.0, 0.0

    if precision_bias_fn is None:
        precision_bias_fn = recall_bias_fn
    if precision_cardinality_fn is None:
        precision_cardinality_fn = recall_cardinality_fn

    if anomaly_ranges is None:
        anomaly_ranges = compute_window_indices(anomalies)
    if prediction_ranges is None:
        prediction_ranges = compute_window_indices(predictions)

    recall = _compute_overlap(
        predictions,
        prediction_ranges,
        anomaly_ranges,
        alpha,
        recall_bias_fn,
        recall_cardinality_fn,
    )
    precision = _compute_overlap(
        anomalies,
        anomaly_ranges,
        prediction_ranges,
        0,
        precision_bias_fn,
        precision_cardinality_fn,
        use_window_weight=weighted_precision,
    )

    return precision, recall




class AnoEvaluator:
    """
    Paper metric mapping:
      F1    = point-wise F1
      F1@K  = PaK-AUC style metric
      F1-C  = composite F1
      F1-R  = range-based F1
    """

    def __init__(self, preds, targets):
        preds = np.asarray(preds).astype(int)
        targets = np.asarray(targets).astype(int)

        assert len(preds) == len(targets), "preds and targets must have same length"

        self.targets = targets
        self.preds = preds

    def eval_naive_f1(self):
        f1 = metrics.f1_score(self.targets, self.preds, zero_division=0)
        prec = metrics.precision_score(self.targets, self.preds, zero_division=0)
        recall = metrics.recall_score(self.targets, self.preds, zero_division=0)
        return f1, prec, recall

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
        recall_pak_auc = metrics.auc(ks, pak_metrics[:, 2]) / 100.0

        return f1_pak_auc, prec_pak_auc, recall_pak_auc

    def eval_f1_composite(self):
        true_events = get_events(self.targets)
        prec_pw, rec_ew, f1_comp = get_composite_fscore_raw(
            self.preds,
            true_events,
            self.targets,
            return_prec_rec=True
        )
        return f1_comp, prec_pw, rec_ew

    def eval_f1_range(self):
        epsilon = 1e-8
        label_ranges = compute_window_indices(self.targets)

        prec, recall = ts_precision_and_recall(
            self.targets,
            self.preds,
            alpha=0,
            anomaly_ranges=label_ranges,
            weighted_precision=True,
        )
        f1 = 2 * prec * recall / (prec + recall + epsilon)
        return f1, prec, recall

    def paper_scores(self):
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


def return_scores(evaluator):
    """
    Keeps a similar API style to your earlier code.
    """
    result_f1, prec, rec = evaluator.eval_naive_f1()
    result_f1_pak, pre_pak, rec_pak = evaluator.eval_pak_auc()
    result_f1_comp, pre_comp, rec_comp = evaluator.eval_f1_composite()
    result_f1_range, pre_range, rec_range = evaluator.eval_f1_range()

    return (
        round(result_f1, 5), round(prec, 5), round(rec, 5),
        round(result_f1_pak, 5), round(pre_pak, 5), round(rec_pak, 5),
        round(result_f1_comp, 5), round(pre_comp, 5), round(rec_comp, 5),
        round(result_f1_range, 5), round(pre_range, 5), round(rec_range, 5)
    )




def evaluate_method_from_scores(
    method_name,
    train_scores,
    test_scores,
    test_labels,
    threshold_method="max_normal",
    contamination=None,
    higher_is_more_anomalous=True,
):
    """
    Evaluate any method from train/test anomaly scores.

    Example:
      - HAD-PCN reactive NLL scores
      - HAD-PCN proactive trajectory scores
      - forecasting baseline anomaly scores
    """
    threshold = threshold_from_train_scores(
        train_scores,
        method=threshold_method,
        contamination=contamination,
    )

    pred_labels = scores_to_labels(
        test_scores,
        threshold,
        higher_is_more_anomalous=higher_is_more_anomalous,
    )

    evaluator = AnoEvaluator(pred_labels, test_labels)
    out = evaluator.paper_scores()
    out["Method"] = method_name
    out["Threshold"] = round(float(threshold), 6)
    return out


def evaluate_multiple_methods(method_dict, test_labels):
    """
    method_dict format:
    {
        "Ours (R)": {
            "train_scores": reactive_train_scores,
            "test_scores": reactive_test_scores,
            "threshold_method": "max_normal",
            "higher_is_more_anomalous": True
        },
        "Ours (P, H=25)": {
            "train_scores": proactive_train_scores,
            "test_scores": proactive_test_scores
        }
    }
    """
    rows = []

    for method_name, cfg in method_dict.items():
        row = evaluate_method_from_scores(
            method_name=method_name,
            train_scores=cfg["train_scores"],
            test_scores=cfg["test_scores"],
            test_labels=test_labels,
            threshold_method=cfg.get("threshold_method", "max_normal"),
            contamination=cfg.get("contamination", None),
            higher_is_more_anomalous=cfg.get("higher_is_more_anomalous", True),
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    cols = ["Method", "F1@K", "F1-C", "F1-R", "F1", "Threshold"]
    return df[cols].sort_values("Method").reset_index(drop=True)




def continual_learning_metrics(task_matrix):
    """
    Compute continual-learning metrics from a task performance matrix A.

    task_matrix[i, j] = performance on task j after training up to task i
    Shape: [num_stages, num_tasks]

    Standard assumptions:
      - diagonal entries A[j, j] are performance right after learning task j
      - last row A[T-1, :] is final performance after all tasks

    Returns:
      Final F1, AA, Forgetting, BWT, FWT
    """
    A = np.asarray(task_matrix, dtype=float)
    n_stages, n_tasks = A.shape

    if n_stages != n_tasks:
        raise ValueError("task_matrix must be square: one row per training stage and one column per task")

    final_f1 = float(np.mean(A[-1, :]))
    aa = float(np.mean(A[np.tril_indices(n_tasks)]))

    forgetting_vals = []
    bwt_vals = []
    fwt_vals = []

    for j in range(n_tasks):
        final_perf_on_j = A[-1, j]
        initial_perf_on_j = A[j, j]

        if j < n_tasks - 1:
            best_after_learning_j = np.max(A[j:, j])
            forgetting_vals.append(best_after_learning_j - final_perf_on_j)
            bwt_vals.append(final_perf_on_j - initial_perf_on_j)

        if j > 0:
            fwt_vals.append(A[j - 1, j])

    forgetting = float(np.mean(forgetting_vals)) if forgetting_vals else 0.0
    bwt = float(np.mean(bwt_vals)) if bwt_vals else 0.0
    fwt = float(np.mean(fwt_vals)) if fwt_vals else 0.0

    return {
        "Final F1": round(final_f1, 5),
        "AA": round(aa, 5),
        "F": round(forgetting, 5),
        "BWT": round(bwt, 5),
        "FWT": round(fwt, 5),
    }



def build_paper_row(method_name, train_scores, test_scores, test_labels,
                    threshold_method="max_normal", contamination=None,
                    higher_is_more_anomalous=True):
    out = evaluate_method_from_scores(
        method_name=method_name,
        train_scores=train_scores,
        test_scores=test_scores,
        test_labels=test_labels,
        threshold_method=threshold_method,
        contamination=contamination,
        higher_is_more_anomalous=higher_is_more_anomalous,
    )
    return [
        out["Method"],
        out["F1@K"],
        out["F1-C"],
        out["F1-R"],
        out["F1"],
    ]


# Example usage
if __name__ == "__main__":
    # Dummy example
    rng = np.random.default_rng(42)

    n_train = 300
    n_test = 500

    train_scores_r = rng.normal(0.2, 0.05, size=n_train)
    test_scores_r = rng.normal(0.2, 0.05, size=n_test)
    test_labels = np.zeros(n_test, dtype=int)

    # Inject anomaly windows
    anomaly_windows = [(100, 130), (250, 280), (400, 430)]
    for s, e in anomaly_windows:
        test_labels[s:e] = 1
        test_scores_r[s:e] += 0.25

    reactive_results = evaluate_method_from_scores(
        method_name="Ours (R)",
        train_scores=train_scores_r,
        test_scores=test_scores_r,
        test_labels=test_labels,
        threshold_method="max_normal",
        higher_is_more_anomalous=True,
    )
    print("Reactive:", reactive_results)

    train_scores_p = rng.normal(0.18, 0.04, size=n_train)
    test_scores_p = rng.normal(0.18, 0.04, size=n_test)
    for s, e in anomaly_windows:
        test_scores_p[max(0, s - 10):e] += 0.20  # earlier warning behavior

    proactive_results = evaluate_method_from_scores(
        method_name="Ours (P, H=25)",
        train_scores=train_scores_p,
        test_scores=test_scores_p,
        test_labels=test_labels,
        threshold_method="max_normal",
        higher_is_more_anomalous=True,
    )
    print("Proactive:", proactive_results)

    method_dict = {
        "Ours (R)": {
            "train_scores": train_scores_r,
            "test_scores": test_scores_r,
        },
        "Ours (P, H=25)": {
            "train_scores": train_scores_p,
            "test_scores": test_scores_p,
        },
    }

    table_df = evaluate_multiple_methods(method_dict, test_labels)
    print("\nPaper-style table:")
    print(table_df)

    # continual-learning matrix
    A = np.array([
        [0.89, 0.00, 0.00, 0.00],
        [0.86, 0.91, 0.00, 0.00],
        [0.84, 0.89, 0.90, 0.00],
        [0.82, 0.87, 0.88, 0.92],
    ])
    cl_metrics = continual_learning_metrics(A)
    print("\nContinual-learning metrics:")
    print(cl_metrics)
