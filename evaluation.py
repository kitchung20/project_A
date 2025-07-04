import numpy as np
import pandas as pd

def convert_timestamp_series_to_epoch(series):
    return (
        (series - pd.Timestamp(year=1970, month=1, day=1)) // pd.Timedelta(seconds=1)
    ).values

def compute_simple_matching_precision_recall_for_one_threshold(
    matching_max_days,
    threshold,
    series_ground_truth_manoeuvre_timestamps,
    series_predictions,
):
    """
    :param matching_max_days
    :param threshold
    :param series_ground_truth_manoeuvre_timestamps
    :param series_predictions: The index of this series should be the timestamps of the predictions.
    :return: (precision, recall)

   Computes the precision and recall at one anomaly threshold.

   Does this using an implementation of the framework proposed by Zhao:
   Zhao, L. (2021). Event prediction in the big data era: A systematic survey. ACM Computing Surveys (CSUR), 54(5), 1-37.
   https://doi.org/10.1145/3450287

   The method matches each manoeuvre prediction with the closest ground-truth manoeuvre, if it is within a time window.

   Predictions with a match are then true positives and those without a match are false positives. Ground-truth manoeuvres
   with no matching prediction are counted as false negatives.
   """

    matching_max_distance_seconds = pd.Timedelta(days=matching_max_days).total_seconds()

    dict_predictions_to_ground_truth = {}
    dict_ground_truth_to_predictions = {}

    manoeuvre_timestamps_seconds = convert_timestamp_series_to_epoch(series_ground_truth_manoeuvre_timestamps)
    pred_time_stamps_seconds = convert_timestamp_series_to_epoch(series_predictions.index)
    predictions = series_predictions.to_numpy()

    for i in range(predictions.shape[0]):
        if predictions[i] >= threshold:
            left_index = np.searchsorted(
                manoeuvre_timestamps_seconds, pred_time_stamps_seconds[i]
            )

            if left_index != 0:
                left_index -= 1

            index_of_closest = left_index

            if (left_index < series_ground_truth_manoeuvre_timestamps.shape[0] - 1) and (
                abs(manoeuvre_timestamps_seconds[left_index] - pred_time_stamps_seconds[i])
                > abs(manoeuvre_timestamps_seconds[left_index + 1] - pred_time_stamps_seconds[i])
            ):
                index_of_closest = left_index + 1

            diff = abs(manoeuvre_timestamps_seconds[index_of_closest] - pred_time_stamps_seconds[i])

            if diff < matching_max_distance_seconds:
                dict_predictions_to_ground_truth[i] = (
                    index_of_closest,
                    diff,
                )
                if index_of_closest in dict_ground_truth_to_predictions:
                    dict_ground_truth_to_predictions[index_of_closest].append(i)
                else:
                    dict_ground_truth_to_predictions[index_of_closest] = [i]

    positive_prediction_indices = np.argwhere(predictions >= threshold)[:, 0]
    list_false_positives = [
        pred_ind for pred_ind in positive_prediction_indices if pred_ind not in dict_predictions_to_ground_truth.keys()
    ]
    list_false_negatives = [
        true_ind for true_ind in np.arange(0, len(series_ground_truth_manoeuvre_timestamps))
        if true_ind not in dict_ground_truth_to_predictions.keys()
    ]

    precision = len(dict_ground_truth_to_predictions) / (len(dict_ground_truth_to_predictions) + len(list_false_positives))
    recall = len(dict_ground_truth_to_predictions) / (len(dict_ground_truth_to_predictions) + len(list_false_negatives))

    return (precision, recall,)



def compute_pr_curve(
    matching_max_days,
    series_ground_truth_manoeuvre_timestamps,
    series_predictions,
    num_thresholds= 50
):
    min_score = series_predictions.min()
    max_score = series_predictions.max()
    thresholds = np.linspace(min_score, max_score, num_thresholds)
    records = []
    for thr in thresholds:
        prec, rec = compute_simple_matching_precision_recall_for_one_threshold(
            matching_max_days, thr,
            series_ground_truth_manoeuvre_timestamps,
            series_predictions
        )
        records.append((thr, prec, rec))
    return pd.DataFrame(records, columns=['threshold', 'precision', 'recall'])