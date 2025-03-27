import time
import sys
import os
import glob
import math
import threading
import concurrent.futures as cf
import random
import re

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Input, Model, layers, metrics, losses, callbacks, optimizers, models, utils
from keras import backend as K
import gc
import keras_tuner as kt
from pyfaidx import Fasta

import sys
import os

@utils.register_keras_serializable()
class CustomBinaryFocalLoss(losses.Loss):
    def __init__(self,
                 dominant_class_index=0,
                 # Dominant class multipliers
                 dominant_correct_multiplier=0.99,    # Reward when dominant class is correct
                 dominant_incorrect_multiplier=2.5,     # Penalty when dominant class is incorrect
                 # Expanded non-dominant multipliers for hard labels
                 other_class_true_positive_multiplier=0.05,   # Reward when y_true==1 and prediction is positive
                 other_class_false_negative_multiplier=3.0,     # Punish when y_true==1 but prediction is negative
                 other_class_false_positive_multiplier=1.0,     # Punish when y_true==0 but prediction is positive
                 other_class_true_negative_multiplier=0.99,     # Reward when y_true==0 and prediction is negative
                 # For smoothed labels (0 < y_true < 1)
                 smoothing_multiplier=0.5,              # Scales the effect of a smoothed label
                 smoothing_as_correct=True,             # If True, a high prediction on a smoothed label is rewarded; else, punished
                 threshold=0.5,                         # Threshold to decide if a prediction is "positive"
                 # Focal loss parameters
                 focal_gamma=2.0,                       # Focusing parameter gamma
                 focal_alpha=0.25,                      # Balance parameter alpha
                 name="custom_binary_focal_loss",
                 reduction="sum_over_batch_size"):
        super().__init__(name=name)
        self.dominant_class_index = dominant_class_index
        self.dominant_correct_multiplier = dominant_correct_multiplier
        self.dominant_incorrect_multiplier = dominant_incorrect_multiplier

        self.other_class_true_positive_multiplier = other_class_true_positive_multiplier
        self.other_class_false_negative_multiplier = other_class_false_negative_multiplier
        self.other_class_false_positive_multiplier = other_class_false_positive_multiplier
        self.other_class_true_negative_multiplier = other_class_true_negative_multiplier

        self.smoothing_multiplier = smoothing_multiplier
        self.smoothing_as_correct = smoothing_as_correct
        self.threshold = threshold

        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha

    def call(self, y_true, y_pred):
        # Prevent log(0) issues.
        epsilon = K.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Reshape to (batch_size, num_classes)
        y_true = tf.reshape(y_true, [-1, tf.shape(y_true)[-1]])
        y_pred = tf.reshape(y_pred, [-1, tf.shape(y_pred)[-1]])
        
        # Compute the focal loss elementwise.
        # For each element, p_t = y_pred if y_true==1, else 1 - y_pred.
        p_t = tf.where(tf.equal(y_true, tf.constant(1.0, dtype=y_true.dtype)), y_pred, 1 - y_pred)
        focal_loss = - self.focal_alpha * tf.pow(1 - p_t, self.focal_gamma) * tf.math.log(p_t)
        
        # Determine the number of classes.
        num_classes = tf.shape(y_true)[1]
        
        # Create masks for the dominant vs. non-dominant classes.
        dominant_mask = tf.one_hot(self.dominant_class_index, depth=num_classes, dtype=tf.float32)
        non_dominant_mask = tf.cast(1.0 - dominant_mask, dtype=tf.float32)
        
        # === Dominant Class Weighting ===
        # For the dominant class, use one multiplier if y_true==1 and another if y_true==0.
        dominant_true = y_true[:, self.dominant_class_index]  # shape: (batch_size,)
        dominant_weight = tf.where(
            tf.equal(dominant_true, tf.constant(1.0, dtype=y_true.dtype)),
            tf.constant(self.dominant_correct_multiplier, dtype=y_true.dtype),
            tf.constant(self.dominant_incorrect_multiplier, dtype=y_true.dtype)
        )
        dominant_weight = tf.expand_dims(dominant_weight, axis=1)  # shape: (batch_size, 1)
        
        # === Non-Dominant Class Weighting ===
        # Distinguish between hard labels (exactly 0 or 1) and smoothed labels (0 < y_true < 1).
        is_hard_positive = tf.equal(y_true, tf.constant(1.0, dtype=y_true.dtype))
        is_hard_negative = tf.equal(y_true, tf.constant(0.0, dtype=y_true.dtype))
        is_hard = tf.logical_or(is_hard_positive, is_hard_negative)
        
        # Determine if the prediction is "positive" (i.e. y_pred >= threshold).
        pred_positive = tf.greater_equal(y_pred, tf.constant(self.threshold, dtype=y_true.dtype))
        
        # For hard labels:
        #   - If y_true==1:
        #       * If prediction is positive: use true positive multiplier.
        #       * Else: use false negative multiplier.
        #   - If y_true==0:
        #       * If prediction is positive: use false positive multiplier.
        #       * Else: use true negative multiplier.
        hard_weight = tf.where(
            tf.equal(y_true, tf.constant(1.0, dtype=y_true.dtype)),
            tf.where(
                pred_positive,
                tf.constant(self.other_class_true_positive_multiplier, dtype=y_true.dtype),
                tf.constant(self.other_class_false_negative_multiplier, dtype=y_true.dtype)
            ),
            tf.where(
                tf.equal(y_true, tf.constant(0.0, dtype=y_true.dtype)),
                tf.where(
                    pred_positive,
                    tf.constant(self.other_class_false_positive_multiplier, dtype=y_true.dtype),
                    tf.constant(self.other_class_true_negative_multiplier, dtype=y_true.dtype)
                ),
                tf.constant(1.0, dtype=y_true.dtype)  # fallback; should not occur for a hard label.
            )
        )
        
        # For smoothed labels: (values strictly between 0 and 1)
        is_smoothed = tf.logical_and(
            tf.greater(y_true, tf.constant(0.0, dtype=y_true.dtype)),
            tf.less(y_true, tf.constant(1.0, dtype=y_true.dtype))
        )
        if self.smoothing_as_correct:
            smoothed_weight = tf.where(
                pred_positive,
                (1.0 - y_true) * self.smoothing_multiplier,  # reward by reducing the loss, smaller reduction for further distance
                1.0 * self.other_class_false_positive_multiplier   # punish for predicting a false positive
            )
        # elif self.smoothing_as_correct == None:
            
        else:
            smoothed_weight = tf.where(
                pred_positive,
                1.0 + (1-y_true) * self.smoothing_multiplier,  # punish, punishment increases with distance
                1.0 * self.other_class_true_negative_multiplier   # reward for predicting a true negative
            )
        
        # Combine weights for non-dominant classes.
        non_dominant_weight = tf.where(
            is_hard,
            hard_weight,
            tf.where(
                is_smoothed,
                smoothed_weight,
                tf.constant(1.0, dtype=y_true.dtype)  # fallback
            )
        )
        
        # Reshape the masks so they broadcast properly.
        dominant_mask = tf.reshape(dominant_mask, tf.stack([tf.constant(1, dtype=tf.int32), num_classes]))
        non_dominant_mask = tf.reshape(non_dominant_mask, tf.stack([tf.constant(1, dtype=tf.int32), num_classes]))
        
        # Combine weights: for each sample and class,
        # use dominant_weight for the dominant class and non_dominant_weight for others.
        weights = dominant_mask * dominant_weight + non_dominant_mask * non_dominant_weight
        
        # Compute the final weighted loss.
        weighted_loss = focal_loss * weights
        return tf.reduce_mean(weighted_loss)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'dominant_class_index': self.dominant_class_index,
            'dominant_correct_multiplier': self.dominant_correct_multiplier,
            'dominant_incorrect_multiplier': self.dominant_incorrect_multiplier,
            'other_class_true_positive_multiplier': self.other_class_true_positive_multiplier,
            'other_class_false_negative_multiplier': self.other_class_false_negative_multiplier,
            'other_class_false_positive_multiplier': self.other_class_false_positive_multiplier,
            'other_class_true_negative_multiplier': self.other_class_true_negative_multiplier,
            'smoothing_multiplier': self.smoothing_multiplier,
            'smoothing_as_correct': self.smoothing_as_correct,
            'threshold': self.threshold,
            'focal_gamma': self.focal_gamma,
            'focal_alpha': self.focal_alpha
        })
        return config
   
    
@utils.register_keras_serializable()
class CustomBinaryCrossentropyLoss(losses.Loss):
    def __init__(self,
                 dominant_class_index=0,
                 # Dominant class multipliers
                 dominant_correct_multiplier=0.99,    # Reward when dominant class is correct
                 dominant_incorrect_multiplier=2.5,     # Penalty when dominant class is incorrect
                 # Non-dominant class multipliers for hard labels
                 other_class_true_positive_multiplier=0.05,   # Reward when y_true==1 and prediction is positive
                 other_class_false_negative_multiplier=3.0,     # Punish when y_true==1 but prediction is negative
                 other_class_false_positive_multiplier=1.0,     # Punish when y_true==0 but prediction is positive
                 other_class_true_negative_multiplier=0.99,     # Reward when y_true==0 and prediction is negative
                 # Focal loss parameters
                 focal_gamma=2.0,                       # Focusing parameter gamma
                 focal_alpha=0.25,                      # Balance parameter alpha
                 # Optional label smoothing (standard method)
                 label_smoothing=0.0,                   # If > 0, perform standard label smoothing on y_true.
                 name="custom_binary_focal_loss",
                 reduction="sum_over_batch_size"):
        super().__init__(name=name)
        self.dominant_class_index = dominant_class_index
        self.dominant_correct_multiplier = dominant_correct_multiplier
        self.dominant_incorrect_multiplier = dominant_incorrect_multiplier

        self.other_class_true_positive_multiplier = other_class_true_positive_multiplier
        self.other_class_false_negative_multiplier = other_class_false_negative_multiplier
        self.other_class_false_positive_multiplier = other_class_false_positive_multiplier
        self.other_class_true_negative_multiplier = other_class_true_negative_multiplier

        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        # Prevent log(0) issues.
        epsilon = K.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Reshape to (batch_size, num_classes)
        y_true = tf.reshape(y_true, [-1, tf.shape(y_true)[-1]])
        y_pred = tf.reshape(y_pred, [-1, tf.shape(y_pred)[-1]])
        
        # Optionally apply standard label smoothing.
        if self.label_smoothing > 0:
            num_classes = tf.cast(tf.shape(y_true)[1], y_true.dtype)
            y_true = y_true * (1.0 - self.label_smoothing) + (self.label_smoothing / num_classes)
        
        # Compute the focal loss elementwise.
        # For each element, p_t = y_pred if y_true==1, else 1 - y_pred.
        p_t = tf.where(tf.equal(y_true, tf.constant(1.0, dtype=y_true.dtype)), y_pred, 1 - y_pred)
        focal_loss = - self.focal_alpha * tf.pow(1 - p_t, self.focal_gamma) * tf.math.log(p_t)
        
        # Determine the number of classes.
        num_classes = tf.shape(y_true)[1]
        
        # Create masks for the dominant vs. non-dominant classes.
        dominant_mask = tf.one_hot(self.dominant_class_index, depth=num_classes, dtype=tf.float32)
        non_dominant_mask = tf.cast(1.0 - dominant_mask, dtype=tf.float32)
        
        # === Dominant Class Weighting ===
        dominant_true = y_true[:, self.dominant_class_index]  # shape: (batch_size,)
        dominant_weight = tf.where(
            tf.equal(dominant_true, tf.constant(1.0, dtype=y_true.dtype)),
            tf.constant(self.dominant_correct_multiplier, dtype=y_true.dtype),
            tf.constant(self.dominant_incorrect_multiplier, dtype=y_true.dtype)
        )
        dominant_weight = tf.expand_dims(dominant_weight, axis=1)  # shape: (batch_size, 1)
        
        # === Non-Dominant Class Weighting ===
        # For binary hard labels:
        #   - If y_true==1:
        #       * Use true positive multiplier if prediction is positive, else false negative multiplier.
        #   - If y_true==0:
        #       * Use false positive multiplier if prediction is positive, else true negative multiplier.
        pred_positive = tf.greater_equal(y_pred, tf.constant(0.5, dtype=y_true.dtype))
        hard_weight = tf.where(
            tf.equal(y_true, tf.constant(1.0, dtype=y_true.dtype)),
            tf.where(
                pred_positive,
                tf.constant(self.other_class_true_positive_multiplier, dtype=y_true.dtype),
                tf.constant(self.other_class_false_negative_multiplier, dtype=y_true.dtype)
            ),
            tf.where(
                tf.equal(y_true, tf.constant(0.0, dtype=y_true.dtype)),
                tf.where(
                    pred_positive,
                    tf.constant(self.other_class_false_positive_multiplier, dtype=y_true.dtype),
                    tf.constant(self.other_class_true_negative_multiplier, dtype=y_true.dtype)
                ),
                tf.constant(1.0, dtype=y_true.dtype)  # Fallback, not expected for binary data.
            )
        )
        non_dominant_weight = hard_weight
        
        # Reshape the masks so they broadcast properly.
        dominant_mask = tf.reshape(dominant_mask, tf.stack([tf.constant(1, dtype=tf.int32), num_classes]))
        non_dominant_mask = tf.reshape(non_dominant_mask, tf.stack([tf.constant(1, dtype=tf.int32), num_classes]))
        
        # Combine weights for each sample and class.
        weights = dominant_mask * dominant_weight + non_dominant_mask * non_dominant_weight
        
        # Compute the final weighted loss.
        weighted_loss = focal_loss * weights
        return tf.reduce_mean(weighted_loss)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'dominant_class_index': self.dominant_class_index,
            'dominant_correct_multiplier': self.dominant_correct_multiplier,
            'dominant_incorrect_multiplier': self.dominant_incorrect_multiplier,
            'other_class_true_positive_multiplier': self.other_class_true_positive_multiplier,
            'other_class_false_negative_multiplier': self.other_class_false_negative_multiplier,
            'other_class_false_positive_multiplier': self.other_class_false_positive_multiplier,
            'other_class_true_negative_multiplier': self.other_class_true_negative_multiplier,
            'focal_gamma': self.focal_gamma,
            'focal_alpha': self.focal_alpha,
            'label_smoothing': self.label_smoothing
        })
        return config