# import time
# import sys
# import os
# import glob
# import math
# import threading
# import concurrent.futures as cf
# import random
# import re

# import numpy as np
# import pandas as pd
import tensorflow as tf
from keras import metrics, utils # Input, Model, layers, losses, callbacks, optimizers, models, 
from keras import backend as K
# import gc
# import keras_tuner as kt
# from pyfaidx import Fasta

# K.clear_session()
# gc.collect()

# datasets_path = "../../Datasets/"
# models_path = "../../Models/"

@utils.register_keras_serializable()
class CustomNoBackgroundF1Score(metrics.Metric):
    def __init__(self, num_classes, average='weighted', threshold=0.5, name='no_background_f1', **kwargs):
        """
        Custom F1 score metric that only considers non-dominant classes (ignoring index 0).
        
        This version is designed for multi-encoded labels where:
          - The dominant class (index 0) is represented as a hard label [1, 0, 0, ...]
          - For non-dominant classes (indices 1 to num_classes-1), only an exact label of 1 is considered positive.
            (Any partial credit/smoothed values below 1 are treated as 0.)
          - Predictions are thresholded (default threshold = 0.5) to decide 1 vs. 0.
        
        Args:
            num_classes (int): Total number of classes.
            average (str): 'weighted' (default) to weight by support or 'macro' for a simple average.
            threshold (float): Threshold on y_pred to decide a positive (default 0.5).
            name (str): Name of the metric.
            **kwargs: Additional keyword arguments.
        """
        super(CustomNoBackgroundF1Score, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.threshold = threshold
        if average not in ['weighted', 'macro']:
            raise ValueError("average must be 'weighted' or 'macro'")
        self.average = average

        # Create state variables to accumulate counts for each class.
        # We use a vector of length num_classes but we will update only indices 1...num_classes-1.
        self.true_positives = self.add_weight(
            name='tp', shape=(num_classes,), initializer='zeros', dtype=tf.float32
        )
        self.false_positives = self.add_weight(
            name='fp', shape=(num_classes,), initializer='zeros', dtype=tf.float32
        )
        self.false_negatives = self.add_weight(
            name='fn', shape=(num_classes,), initializer='zeros', dtype=tf.float32
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Updates the metric state.
        
        Args:
            y_true: Tensor of shape (batch_size, num_classes). These are multi-encoded labels.
                    For non-dominant classes, a label is considered positive only if it is exactly 1.
            y_pred: Tensor of shape (batch_size, num_classes) with predictions (e.g. probabilities).
            sample_weight: Optional sample weights.
        """
        
        # Flatten all dimensions except the last one (which should be num_classes).
        y_true = tf.reshape(y_true, [-1, self.num_classes])
        y_pred = tf.reshape(y_pred, [-1, self.num_classes])
        
        # We want to ignore the dominant class (index 0) and work on classes 1...num_classes-1.
        # Assume y_true and y_pred are both of shape (batch_size, num_classes).
        y_true_non_dominant = y_true[:, 1:]
        y_pred_non_dominant = y_pred[:, 1:]
        
        # For ground truth: treat a class as positive only if its value is exactly 1.
        one_value = tf.cast(1.0, dtype=y_true_non_dominant.dtype)
        y_true_bin = tf.cast(tf.equal(y_true_non_dominant, one_value), tf.int32)
        # For predictions: apply thresholding.
        y_pred_bin = tf.cast(y_pred_non_dominant >= self.threshold, tf.int32)
        
        # (Optionally) apply sample weighting.
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.int32)
            sample_weight = tf.reshape(sample_weight, (-1, 1))
            y_true_bin = y_true_bin * sample_weight
            y_pred_bin = y_pred_bin * sample_weight
        
        # Compute per-class true positives, false positives, and false negatives for non-dominant classes.
        tp = tf.reduce_sum(tf.cast(y_true_bin * y_pred_bin, tf.float32), axis=0)
        fp = tf.reduce_sum(tf.cast((1 - y_true_bin) * y_pred_bin, tf.float32), axis=0)
        fn = tf.reduce_sum(tf.cast(y_true_bin * (1 - y_pred_bin), tf.float32), axis=0)
        
        # Our state variables have length num_classes. We want to update only indices 1... with our computed values.
        zeros = tf.zeros([1], dtype=tf.float32)
        tp_update = tf.concat([zeros, tp], axis=0)
        fp_update = tf.concat([zeros, fp], axis=0)
        fn_update = tf.concat([zeros, fn], axis=0)
        
        self.true_positives.assign_add(tp_update)
        self.false_positives.assign_add(fp_update)
        self.false_negatives.assign_add(fn_update)

    def result(self):
        """
        Computes the F1 score over the non-dominant classes (indices 1...num_classes-1).
        """
        # Select non-dominant classes only.
        tp = self.true_positives[1:]
        fp = self.false_positives[1:]
        fn = self.false_negatives[1:]
        
        precision = tf.math.divide_no_nan(tp, tp + fp)
        recall = tf.math.divide_no_nan(tp, tp + fn)
        f1 = tf.math.divide_no_nan(2 * precision * recall, precision + recall)
        
        if self.average == 'weighted':
            support = tp + fn
            weighted_f1 = tf.reduce_sum(f1 * support) / (tf.reduce_sum(support) + K.epsilon())
            return weighted_f1
        else:  # macro
            return tf.reduce_mean(f1)

    def reset_states(self):
        """
        Resets all of the metric state variables.
        """
        for v in self.variables:
            v.assign(tf.zeros_like(v))
            
    def get_config(self):
        """
        Returns the configuration of the metric, so it can be recreated later.
        """
        config = super(CustomNoBackgroundF1Score, self).get_config()
        config.update({
            'num_classes': self.num_classes,
            'average': self.average,
            'threshold': self.threshold,
        })
        return config

@utils.register_keras_serializable()
class CustomConditionalF1Score(metrics.Metric):
    def __init__(self, threshold=0.5, average='weighted', filter_mode='either', name='conditional_f1', **kwargs):
        """
        Custom F1 score metric that computes the F1 score only for target columns (columns 1-4).
        Additionally, only rows meeting a filtering criterion are included in the calculation.
        
        Args:
            threshold (float): Threshold on y_pred to decide a positive (default = 0.5).
            average (str): 'weighted' (default) to weight by support or 'macro' for a simple average.
            filter_mode (str): Determines which rows to include based on the target columns.
                               Options:
                                  - 'pred': Only rows where y_pred (after thresholding) has at least one 1.
                                  - 'true': Only rows where y_true (exactly equal to 1) has at least one 1.
                                  - 'either': Rows where either y_true or y_pred has at least one 1.
            name (str): Name of the metric.
            **kwargs: Additional keyword arguments.
        
        Note:
            This metric only tracks columns 1-4 (0-indexed). Column 0 (the dominant background class)
            is ignored completely.
        """
        metric_name = f'{name}_{filter_mode}'
        
        super(CustomConditionalF1Score, self).__init__(name=metric_name, **kwargs)
        self.threshold = threshold
        if average not in ['weighted', 'macro']:
            raise ValueError("average must be 'weighted' or 'macro'")
        self.average = average
        
        if filter_mode not in ['pred', 'true', 'either']:
            raise ValueError("filter_mode must be 'pred', 'true', or 'either'")
        self.filter_mode = filter_mode
        
        # We are tracking only 4 target columns (columns 1 to 4).
        self.num_target_columns = 4
        self.true_positives = self.add_weight(
            name='tp', shape=(self.num_target_columns,), initializer='zeros', dtype=tf.float32
        )
        self.false_positives = self.add_weight(
            name='fp', shape=(self.num_target_columns,), initializer='zeros', dtype=tf.float32
        )
        self.false_negatives = self.add_weight(
            name='fn', shape=(self.num_target_columns,), initializer='zeros', dtype=tf.float32
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Reshape inputs so that the last dimension is the number of classes.
        y_true = tf.reshape(y_true, [-1, tf.shape(y_true)[-1]])
        y_pred = tf.reshape(y_pred, [-1, tf.shape(y_pred)[-1]])
        
        # Only consider columns 1-4 (ignoring index 0).
        y_true_subset = y_true[:, 1:5]
        y_pred_subset = y_pred[:, 1:5]
        
        # For ground truth, treat a label as positive only if its value is exactly 1.
        y_true_bin = tf.cast(tf.equal(y_true_subset, 1.0), tf.int32)
        # For predictions, apply the threshold to decide 1 vs. 0.
        y_pred_bin = tf.cast(y_pred_subset >= self.threshold, tf.int32)
        
        # Compute a row-level mask based on the filter_mode.
        if self.filter_mode == 'pred':
            mask = tf.reduce_any(tf.equal(y_pred_bin, 1), axis=1)
        elif self.filter_mode == 'true':
            mask = tf.reduce_any(tf.equal(y_true_bin, 1), axis=1)
        else:  # 'either'
            mask = tf.logical_or(
                tf.reduce_any(tf.equal(y_pred_bin, 1), axis=1),
                tf.reduce_any(tf.equal(y_true_bin, 1), axis=1)
            )
        
        # Apply the mask so only selected rows are used for the metric update.
        y_true_filtered = tf.boolean_mask(y_true_bin, mask)
        y_pred_filtered = tf.boolean_mask(y_pred_bin, mask)
        
        # Optionally apply sample weighting.
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32)
            sample_weight = tf.reshape(sample_weight, [-1, 1])
            y_true_filtered = y_true_filtered * sample_weight
            y_pred_filtered = y_pred_filtered * sample_weight
        
        # Compute per-column true positives, false positives, and false negatives.
        tp = tf.reduce_sum(tf.cast(y_true_filtered * y_pred_filtered, tf.float32), axis=0)
        fp = tf.reduce_sum(tf.cast((1 - y_true_filtered) * y_pred_filtered, tf.float32), axis=0)
        fn = tf.reduce_sum(tf.cast(y_true_filtered * (1 - y_pred_filtered), tf.float32), axis=0)
        
        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)

    def result(self):
        precision = tf.math.divide_no_nan(self.true_positives, self.true_positives + self.false_positives)
        recall = tf.math.divide_no_nan(self.true_positives, self.true_positives + self.false_negatives)
        f1 = tf.math.divide_no_nan(2 * precision * recall, precision + recall)
        
        if self.average == 'weighted':
            support = self.true_positives + self.false_negatives
            return tf.reduce_sum(f1 * support) / (tf.reduce_sum(support) + K.epsilon())
        else:  # 'macro'
            return tf.reduce_mean(f1)

    def reset_states(self):
        for v in self.variables:
            v.assign(tf.zeros_like(v))
            
    def get_config(self):
        config = super(CustomConditionalF1Score, self).get_config()
        config.update({
            'threshold': self.threshold,
            'average': self.average,
            'filter_mode': self.filter_mode,
        })
        return config

@utils.register_keras_serializable()
class CustomFalsePositiveDistance(metrics.Metric):
    def __init__(self, num_classes, threshold=0.5, window=100, name='false_positive_distance', **kwargs):
        """
        Metric that accumulates a running average “distance” error for false positive predictions,
        ignoring the dominant (background) class (index 0).

        For each false positive (i.e. a prediction >= threshold when the strict label is not 1),
        the distance is computed from the raw label value (which encodes proximity to an actual annotation)
        as follows:

            distance = 1 + ((max_credit - v) * (window / max_credit))

        where:
            - v is the raw label value at that position,
            - max_credit is the maximum smoothing credit (0.5 in our scheme), so that if v == 0.5 the distance is 1,
              and if v == 0 the distance is 1 + window (i.e. 101 for window=100).

        Args:
            num_classes (int): Total number of classes.
            threshold (float): Threshold on y_pred to decide a positive.
            window (int): Window size used in the smoothing scheme.
            name (str): Name of the metric.
        """
        super(CustomFalsePositiveDistance, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.threshold = threshold
        self.window = float(window)
        self.max_credit = 0.5  # Based on smoothing scheme.

        # State variables to accumulate total distance and count of false positives.
        self.total_distance = self.add_weight(
            name='total_distance', initializer='zeros', dtype=tf.float32
        )
        self.false_positive_count = self.add_weight(
            name='false_positive_count', initializer='zeros', dtype=tf.float32
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        For non-dominant classes (indices 1:), this method:
          - thresholds predictions,
          - identifies false positives (prediction is positive while strict label != 1),
          - computes the distance error from the raw (smoothed) label value, and
          - accumulates the sum of distances and count of false positives.
        """
        # Ensure shape (batch_size, num_classes)
        y_true = tf.reshape(y_true, [-1, self.num_classes])
        y_pred = tf.reshape(y_pred, [-1, self.num_classes])

        # Ignore the dominant/background class (index 0)
        y_true_non = y_true[:, 1:]
        y_pred_non = y_pred[:, 1:]

        # Threshold predictions
        y_pred_bin = tf.cast(y_pred_non >= self.threshold, tf.float32)

        # For strict classification, a label is positive only if it is exactly 1.
        # So a false positive is when y_pred_bin==1 but y_true (strict) is not 1.
        # (This is similar to the F1 metric, i.e. smoothing values are treated as negatives.)
        false_positive_mask = tf.logical_and(
            tf.equal(y_pred_bin, 1.0),
            tf.not_equal(y_true_non, 1.0)
        )
        false_positive_mask = tf.cast(false_positive_mask, tf.float32)

        # Compute distance per element.
        # In our smoothing scheme:
        #   - At a true annotation (v = 1), we wouldn’t count a false positive.
        #   - In a smoothed region, the maximum credit is 0.5.
        #   - We define:
        #       distance = 1 + ((max_credit - v) * (window / max_credit))
        #     so that if v == 0.5, distance = 1, and if v == 0, distance = 1 + window.
        distance = 1.0 + (self.max_credit - y_true_non) * (self.window / self.max_credit)
        distance = tf.where(distance >= 101.0, tf.constant(125.0, dtype=distance.dtype), distance)

        # Only include entries that are false positives.
        false_positive_distance = distance * false_positive_mask

        # Sum distances and count false positives.
        sum_distance = tf.reduce_sum(false_positive_distance)
        count = tf.reduce_sum(false_positive_mask)

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32)
            sample_weight = tf.reshape(sample_weight, [-1, 1])
            sum_distance = tf.reduce_sum(false_positive_distance * sample_weight)
            count = tf.reduce_sum(false_positive_mask * sample_weight)

        self.total_distance.assign_add(sum_distance)
        self.false_positive_count.assign_add(count)

    def result(self):
        """Returns the average distance error over all false positives (or 0 if none)."""
        return tf.math.divide_no_nan(self.total_distance, self.false_positive_count)

    def reset_states(self):
        """Resets the accumulated total distance and count."""
        self.total_distance.assign(0.0)
        self.false_positive_count.assign(0.0)

    def get_config(self):
        config = super(CustomFalsePositiveDistance, self).get_config()
        config.update({
            'num_classes': self.num_classes,
            'threshold': self.threshold,
            'window': self.window,
        })
        return config


@utils.register_keras_serializable()
class CustomNoBackgroundAUC(metrics.Metric):
    def __init__(self, curve='PR', name='no_background_auc', **kwargs):
        """
        Custom AUC metric computed only for columns 1-4.

        Args:
            curve (str): The type of AUC curve to use, e.g. 'ROC' (default) or 'PR'.
            name (str): Name of the metric.
            **kwargs: Additional keyword arguments.
        """
        super(CustomNoBackgroundAUC, self).__init__(name=name, **kwargs)
        # Store the curve parameter as a string to aid serialization.
        self.curve = curve  
        # Create one AUC metric per target column (columns 1-4).
        self.auc_metrics = [
            metrics.AUC(curve=self.curve, name=f'auc_col_{i+1}')
            for i in range(4)
        ]

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Ensure inputs are 2D tensors with shape (batch_size, num_classes).
        y_true = tf.reshape(y_true, [-1, tf.shape(y_true)[-1]])
        y_pred = tf.reshape(y_pred, [-1, tf.shape(y_pred)[-1]])
        # Select target columns (1-4) and ignore background (column 0).
        y_true_subset = y_true[:, 1:5]
        y_pred_subset = y_pred[:, 1:5]
        # For each target column, update the corresponding AUC metric.
        for i, auc_metric in enumerate(self.auc_metrics):
            # Ground truth: positive only if exactly equal to 1.
            y_true_col = tf.cast(tf.equal(y_true_subset[:, i], 1.0), tf.float32)
            y_pred_col = y_pred_subset[:, i]
            if sample_weight is not None:
                sample_weight = tf.reshape(sample_weight, [-1])
                auc_metric.update_state(y_true_col, y_pred_col, sample_weight=sample_weight)
            else:
                auc_metric.update_state(y_true_col, y_pred_col)

    def result(self):
        # Average AUC over all target columns.
        auc_results = [auc_metric.result() for auc_metric in self.auc_metrics]
        return tf.reduce_mean(auc_results)

    def reset_states(self):
        for auc_metric in self.auc_metrics:
            auc_metric.reset_states()

    def get_config(self):
        config = super(CustomNoBackgroundAUC, self).get_config()
        # Return the curve as a string.
        config.update({
            'curve': self.curve,
        })
        return config


@utils.register_keras_serializable()
class CustomNoBackgroundAccuracy(metrics.Metric):
    def __init__(self, threshold=0.5, name='no_background_accuracy', **kwargs):
        """
        Custom accuracy metric computed only for columns 1-4.

        Args:
            threshold (float): Threshold for y_pred (default 0.5).
            name (str): Name of the metric.
            **kwargs: Additional keyword arguments.
        """
        super(CustomNoBackgroundAccuracy, self).__init__(name=name, **kwargs)
        self.threshold = threshold
        self.total_correct = self.add_weight(name='total_correct', initializer='zeros', dtype=tf.float32)
        self.total_count = self.add_weight(name='total_count', initializer='zeros', dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Reshape inputs to 2D tensors.
        y_true = tf.reshape(y_true, [-1, tf.shape(y_true)[-1]])
        y_pred = tf.reshape(y_pred, [-1, tf.shape(y_pred)[-1]])
        # Extract columns 1-4.
        y_true_subset = y_true[:, 1:5]
        y_pred_subset = y_pred[:, 1:5]
        # Binarize ground truth: positive if exactly 1.
        y_true_bin = tf.cast(tf.equal(y_true_subset, 1.0), tf.int32)
        # Binarize predictions using the threshold.
        y_pred_bin = tf.cast(y_pred_subset >= self.threshold, tf.int32)
        # Element-wise correctness.
        correct = tf.cast(tf.equal(y_true_bin, y_pred_bin), tf.float32)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32)
            # Tile sample weights to match the shape of correct.
            sample_weight = tf.tile(sample_weight, [1, tf.shape(correct)[1]])
            correct = correct * sample_weight
            count = tf.reduce_sum(sample_weight)
        else:
            count = tf.cast(tf.size(correct), tf.float32)
        self.total_correct.assign_add(tf.reduce_sum(correct))
        self.total_count.assign_add(count)

    def result(self):
        return tf.math.divide_no_nan(self.total_correct, self.total_count)

    def reset_states(self):
        self.total_correct.assign(0)
        self.total_count.assign(0)

    def get_config(self):
        config = super(CustomNoBackgroundAccuracy, self).get_config()
        config.update({'threshold': self.threshold})
        return config


@utils.register_keras_serializable()
class CustomNoBackgroundPrecision(metrics.Metric):
    def __init__(self, threshold=0.5, average='weighted', name='no_background_precision', **kwargs):
        """
        Custom precision metric computed only for columns 1-4.

        Args:
            threshold (float): Threshold for y_pred (default 0.5).
            average (str): 'weighted' (default) or 'macro'.
            name (str): Name of the metric.
            **kwargs: Additional keyword arguments.
        """
        super(CustomNoBackgroundPrecision, self).__init__(name=name, **kwargs)
        self.threshold = threshold
        if average not in ['weighted', 'macro']:
            raise ValueError("average must be 'weighted' or 'macro'")
        self.average = average
        self.num_target_columns = 4
        self.true_positives = self.add_weight(
            name='tp', shape=(self.num_target_columns,), initializer='zeros', dtype=tf.float32
        )
        self.false_positives = self.add_weight(
            name='fp', shape=(self.num_target_columns,), initializer='zeros', dtype=tf.float32
        )
        # For weighted averaging, we also need the support (true positives + false negatives).
        self.false_negatives = self.add_weight(
            name='fn', shape=(self.num_target_columns,), initializer='zeros', dtype=tf.float32
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Reshape inputs.
        y_true = tf.reshape(y_true, [-1, tf.shape(y_true)[-1]])
        y_pred = tf.reshape(y_pred, [-1, tf.shape(y_pred)[-1]])
        # Extract target columns (1-4).
        y_true_subset = y_true[:, 1:5]
        y_pred_subset = y_pred[:, 1:5]
        # Binarize ground truth and predictions.
        y_true_bin = tf.cast(tf.equal(y_true_subset, 1.0), tf.int32)
        y_pred_bin = tf.cast(y_pred_subset >= self.threshold, tf.int32)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32)
            sample_weight = tf.tile(sample_weight, [1, tf.shape(y_true_bin)[1]])
            y_true_bin = y_true_bin * tf.cast(sample_weight, tf.int32)
            y_pred_bin = y_pred_bin * tf.cast(sample_weight, tf.int32)
        # Compute counts per column.
        tp = tf.reduce_sum(tf.cast(y_true_bin * y_pred_bin, tf.float32), axis=0)
        fp = tf.reduce_sum(tf.cast((1 - y_true_bin) * y_pred_bin, tf.float32), axis=0)
        fn = tf.reduce_sum(tf.cast(y_true_bin * (1 - y_pred_bin), tf.float32), axis=0)
        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)

    def result(self):
        # Precision: TP / (TP + FP)
        precision = tf.math.divide_no_nan(self.true_positives, self.true_positives + self.false_positives)
        if self.average == 'weighted':
            # Weight each column by its support (TP + FN).
            support = self.true_positives + self.false_negatives
            weighted_precision = tf.reduce_sum(precision * support) / (tf.reduce_sum(support) + K.epsilon())
            return weighted_precision
        else:  # macro
            return tf.reduce_mean(precision)

    def reset_states(self):
        self.true_positives.assign(tf.zeros_like(self.true_positives))
        self.false_positives.assign(tf.zeros_like(self.false_positives))
        self.false_negatives.assign(tf.zeros_like(self.false_negatives))

    def get_config(self):
        config = super(CustomNoBackgroundPrecision, self).get_config()
        config.update({
            'threshold': self.threshold,
            'average': self.average,
        })
        return config


@utils.register_keras_serializable()
class CustomNoBackgroundRecall(metrics.Metric):
    def __init__(self, threshold=0.5, average='weighted', name='no_background_recall', **kwargs):
        """
        Custom recall metric computed only for columns 1-4.

        Args:
            threshold (float): Threshold for y_pred (default 0.5).
            average (str): 'weighted' (default) or 'macro'.
            name (str): Name of the metric.
            **kwargs: Additional keyword arguments.
        """
        super(CustomNoBackgroundRecall, self).__init__(name=name, **kwargs)
        self.threshold = threshold
        if average not in ['weighted', 'macro']:
            raise ValueError("average must be 'weighted' or 'macro'")
        self.average = average
        self.num_target_columns = 4
        self.true_positives = self.add_weight(
            name='tp', shape=(self.num_target_columns,), initializer='zeros', dtype=tf.float32
        )
        self.false_negatives = self.add_weight(
            name='fn', shape=(self.num_target_columns,), initializer='zeros', dtype=tf.float32
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Reshape inputs.
        y_true = tf.reshape(y_true, [-1, tf.shape(y_true)[-1]])
        y_pred = tf.reshape(y_pred, [-1, tf.shape(y_pred)[-1]])
        # Extract target columns (1-4).
        y_true_subset = y_true[:, 1:5]
        y_pred_subset = y_pred[:, 1:5]
        # Binarize ground truth and predictions.
        y_true_bin = tf.cast(tf.equal(y_true_subset, 1.0), tf.int32)
        y_pred_bin = tf.cast(y_pred_subset >= self.threshold, tf.int32)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32)
            sample_weight = tf.tile(sample_weight, [1, tf.shape(y_true_bin)[1]])
            y_true_bin = y_true_bin * tf.cast(sample_weight, tf.int32)
            y_pred_bin = y_pred_bin * tf.cast(sample_weight, tf.int32)
        # Compute per-column true positives and false negatives.
        tp = tf.reduce_sum(tf.cast(y_true_bin * y_pred_bin, tf.float32), axis=0)
        fn = tf.reduce_sum(tf.cast(y_true_bin * (1 - y_pred_bin), tf.float32), axis=0)
        self.true_positives.assign_add(tp)
        self.false_negatives.assign_add(fn)

    def result(self):
        # Recall: TP / (TP + FN)
        recall = tf.math.divide_no_nan(self.true_positives, self.true_positives + self.false_negatives)
        if self.average == 'weighted':
            support = self.true_positives + self.false_negatives
            weighted_recall = tf.reduce_sum(recall * support) / (tf.reduce_sum(support) + K.epsilon())
            return weighted_recall
        else:
            return tf.reduce_mean(recall)

    def reset_states(self):
        self.true_positives.assign(tf.zeros_like(self.true_positives))
        self.false_negatives.assign(tf.zeros_like(self.false_negatives))

    def get_config(self):
        config = super(CustomNoBackgroundRecall, self).get_config()
        config.update({
            'threshold': self.threshold,
            'average': self.average,
        })
        return config


@utils.register_keras_serializable()
class CustomBackgroundOnlyF1Score(metrics.Metric):
    def __init__(self, num_classes, average='weighted', threshold=0.5, name='background_only_f1', **kwargs):
        """
        Custom F1 score metric that only considers the dominant (background) class (index 0).

        This metric is designed for multi-encoded labels where:
          - The dominant class (index 0) aka background is represented as a hard label [1, 0, 0, ...].
          - For the dominant class, a label is considered positive only if it is exactly 1.
          - Predictions are thresholded (default threshold = 0.5) to decide 1 vs. 0.

        Args:
            num_classes (int): Total number of classes.
            average (str): 'weighted' (default) or 'macro'. (Since only one class is considered, this
                           choice won’t make much difference.)
            threshold (float): Threshold on y_pred to decide a positive (default 0.5).
            name (str): Name of the metric.
            **kwargs: Additional keyword arguments.
        """
        super(CustomBackgroundOnlyF1Score, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.threshold = threshold
        if average not in ['weighted', 'macro']:
            raise ValueError("average must be 'weighted' or 'macro'")
        self.average = average

        # We still create vectors of length num_classes, but will only update index 0.
        self.true_positives = self.add_weight(
            name='tp', shape=(num_classes,), initializer='zeros', dtype=tf.float32
        )
        self.false_positives = self.add_weight(
            name='fp', shape=(num_classes,), initializer='zeros', dtype=tf.float32
        )
        self.false_negatives = self.add_weight(
            name='fn', shape=(num_classes,), initializer='zeros', dtype=tf.float32
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Updates the metric state using only the dominant class (index 0).

        Args:
            y_true: Tensor of shape (batch_size, num_classes). For the dominant class,
                    a label is considered positive only if it is exactly 1.
            y_pred: Tensor of shape (batch_size, num_classes) (e.g. probabilities).
            sample_weight: Optional sample weights.
        """
        # Reshape to (-1, num_classes) in case additional dimensions exist.
        y_true = tf.reshape(y_true, [-1, self.num_classes])
        y_pred = tf.reshape(y_pred, [-1, self.num_classes])

        # Extract the dominant class (index 0)
        y_true_dominant = y_true[:, 0]
        y_pred_dominant = y_pred[:, 0]

        # For ground truth, treat as positive only if exactly equal to 1.
        one_value = tf.cast(1.0, dtype=y_true_dominant.dtype)
        y_true_bin = tf.cast(tf.equal(y_true_dominant, one_value), tf.float32)

        # For predictions, apply thresholding.
        y_pred_bin = tf.cast(y_pred_dominant >= self.threshold, tf.float32)

        # Optionally apply sample weighting.
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32)
            sample_weight = tf.reshape(sample_weight, [-1])
            y_true_bin = y_true_bin * sample_weight
            y_pred_bin = y_pred_bin * sample_weight

        # Compute true positives, false positives, and false negatives for the dominant class.
        tp = tf.reduce_sum(y_true_bin * y_pred_bin)
        fp = tf.reduce_sum((1 - y_true_bin) * y_pred_bin)
        fn = tf.reduce_sum(y_true_bin * (1 - y_pred_bin))

        # We create update vectors that place the computed scalar at index 0 and zeros elsewhere.
        zeros = tf.zeros([self.num_classes - 1], dtype=tf.float32)
        tp_update = tf.concat([[tp], zeros], axis=0)
        fp_update = tf.concat([[fp], zeros], axis=0)
        fn_update = tf.concat([[fn], zeros], axis=0)

        self.true_positives.assign_add(tp_update)
        self.false_positives.assign_add(fp_update)
        self.false_negatives.assign_add(fn_update)

    def result(self):
        """
        Computes the F1 score for the dominant (background) class (index 0).
        """
        tp = self.true_positives[0]
        fp = self.false_positives[0]
        fn = self.false_negatives[0]

        precision = tf.math.divide_no_nan(tp, tp + fp)
        recall = tf.math.divide_no_nan(tp, tp + fn)
        f1 = tf.math.divide_no_nan(2 * precision * recall, precision + recall)

        # Although averaging is not critical with a single class, we mirror the interface.
        if self.average == 'weighted':
            support = tp + fn
            weighted_f1 = tf.math.divide_no_nan(f1 * support, support + K.epsilon())
            return weighted_f1
        else:  # macro
            return f1

    def reset_states(self):
        """
        Resets all of the metric state variables.
        """
        for v in self.variables:
            v.assign(tf.zeros_like(v))
            
    def get_config(self):
        """
        Returns the configuration of the metric, so it can be recreated later.
        """
        config = super(CustomBackgroundOnlyF1Score, self).get_config()
        config.update({
            'num_classes': self.num_classes,
            'average': self.average,
            'threshold': self.threshold,
        })
        return config
    
    
custom_metrics_classes = [
    CustomNoBackgroundF1Score, 
    CustomConditionalF1Score, 
    CustomFalsePositiveDistance, 
    CustomNoBackgroundAUC, 
    CustomNoBackgroundAccuracy, 
    CustomNoBackgroundPrecision, 
    CustomNoBackgroundRecall, 
    CustomBackgroundOnlyF1Score
]

def main():
    for item in custom_metrics_classes:
        print(str(item))
        
if __name__ == "__main__":
    main()