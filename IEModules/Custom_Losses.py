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

from .config import DATA_DIR, LOG_DIR, MODEL_DIR, MODULE_DIR, NOTEBOOK_DIR

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
                 background_removed=False,              # New flag: if True, the background (dominant) class is not present
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
        
        self.background_removed = background_removed

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
        
        # Depending on whether the background class is present or not,
        # use either the full (dominant + non-dominant) weighting or only non-dominant weighting.
        if self.background_removed:
            # Only non-dominant weighting is applied to all classes.
            pred_positive = tf.greater_equal(y_pred, tf.constant(self.threshold, dtype=y_true.dtype))
            is_hard_positive = tf.equal(y_true, tf.constant(1.0, dtype=y_true.dtype))
            is_hard_negative = tf.equal(y_true, tf.constant(0.0, dtype=y_true.dtype))
            is_hard = tf.logical_or(is_hard_positive, is_hard_negative)
            
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
                    tf.constant(1.0, dtype=y_true.dtype)
                )
            )
            # For smoothed labels:
            is_smoothed = tf.logical_and(
                tf.greater(y_true, tf.constant(0.0, dtype=y_true.dtype)),
                tf.less(y_true, tf.constant(1.0, dtype=y_true.dtype))
            )
            if self.smoothing_as_correct:
                smoothed_weight = tf.where(
                    pred_positive,
                    (1.0 - y_true) * self.smoothing_multiplier,
                    1.0 * self.other_class_false_positive_multiplier
                )
            else:
                smoothed_weight = tf.where(
                    pred_positive,
                    1.0 + (1.0 - y_true) * self.smoothing_multiplier,
                    1.0 * self.other_class_true_negative_multiplier
                )
            non_dominant_weight = tf.where(
                is_hard,
                hard_weight,
                tf.where(is_smoothed, smoothed_weight, tf.constant(1.0, dtype=y_true.dtype))
            )
            weights = non_dominant_weight
        else:
            # === Dominant Class Weighting ===
            dominant_mask = tf.one_hot(self.dominant_class_index, depth=num_classes, dtype=tf.float32)
            non_dominant_mask = tf.cast(1.0 - dominant_mask, dtype=tf.float32)
            
            dominant_true = y_true[:, self.dominant_class_index]  # shape: (batch_size,)
            dominant_weight = tf.where(
                tf.equal(dominant_true, tf.constant(1.0, dtype=y_true.dtype)),
                tf.constant(self.dominant_correct_multiplier, dtype=y_true.dtype),
                tf.constant(self.dominant_incorrect_multiplier, dtype=y_true.dtype)
            )
            dominant_weight = tf.expand_dims(dominant_weight, axis=1)  # shape: (batch_size, 1)
            
            # === Non-Dominant Class Weighting ===
            pred_positive = tf.greater_equal(y_pred, tf.constant(self.threshold, dtype=y_true.dtype))
            is_hard_positive = tf.equal(y_true, tf.constant(1.0, dtype=y_true.dtype))
            is_hard_negative = tf.equal(y_true, tf.constant(0.0, dtype=y_true.dtype))
            is_hard = tf.logical_or(is_hard_positive, is_hard_negative)
            
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
                    tf.constant(1.0, dtype=y_true.dtype)
                )
            )
            is_smoothed = tf.logical_and(
                tf.greater(y_true, tf.constant(0.0, dtype=y_true.dtype)),
                tf.less(y_true, tf.constant(1.0, dtype=y_true.dtype))
            )
            if self.smoothing_as_correct:
                smoothed_weight = tf.where(
                    pred_positive,
                    (1.0 - y_true) * self.smoothing_multiplier,
                    1.0 * self.other_class_false_positive_multiplier
                )
            else:
                smoothed_weight = tf.where(
                    pred_positive,
                    1.0 + (1.0 - y_true) * self.smoothing_multiplier,
                    1.0 * self.other_class_true_negative_multiplier
                )
            non_dominant_weight = tf.where(
                is_hard,
                hard_weight,
                tf.where(is_smoothed, smoothed_weight, tf.constant(1.0, dtype=y_true.dtype))
            )
            
            dominant_mask = tf.reshape(dominant_mask, tf.stack([tf.constant(1, dtype=tf.int32), num_classes]))
            non_dominant_mask = tf.reshape(non_dominant_mask, tf.stack([tf.constant(1, dtype=tf.int32), num_classes]))
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
            'focal_alpha': self.focal_alpha,
            'background_removed': self.background_removed
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
                 background_removed=False,              # New flag: if True, background class is not present.
                 name="custom_binary_crossentropy_loss",
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
        self.background_removed = background_removed

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
        p_t = tf.where(tf.equal(y_true, tf.constant(1.0, dtype=y_true.dtype)), y_pred, 1 - y_pred)
        focal_loss = - self.focal_alpha * tf.pow(1 - p_t, self.focal_gamma) * tf.math.log(p_t)
        
        num_classes = tf.shape(y_true)[1]
        
        if self.background_removed:
            # Use only non-dominant (binary) weighting for all classes.
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
                    tf.constant(1.0, dtype=y_true.dtype)
                )
            )
            weights = hard_weight
        else:
            dominant_mask = tf.one_hot(self.dominant_class_index, depth=num_classes, dtype=tf.float32)
            non_dominant_mask = tf.cast(1.0 - dominant_mask, dtype=tf.float32)
            
            dominant_true = y_true[:, self.dominant_class_index]
            dominant_weight = tf.where(
                tf.equal(dominant_true, tf.constant(1.0, dtype=y_true.dtype)),
                tf.constant(self.dominant_correct_multiplier, dtype=y_true.dtype),
                tf.constant(self.dominant_incorrect_multiplier, dtype=y_true.dtype)
            )
            dominant_weight = tf.expand_dims(dominant_weight, axis=1)
            
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
                    tf.constant(1.0, dtype=y_true.dtype)
                )
            )
            non_dominant_weight = hard_weight
            
            dominant_mask = tf.reshape(dominant_mask, tf.stack([tf.constant(1, dtype=tf.int32), num_classes]))
            non_dominant_mask = tf.reshape(non_dominant_mask, tf.stack([tf.constant(1, dtype=tf.int32), num_classes]))
            weights = dominant_mask * dominant_weight + non_dominant_mask * non_dominant_weight
        
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
            'label_smoothing': self.label_smoothing,
            'background_removed': self.background_removed
        })
        return config
 

@utils.register_keras_serializable 
class SwitchingFocalLoss(Loss):
    def __init__(self,
                 swap_epoch=3,
                 **shared_kwargs):
        super().__init__(name="switching_focal_loss")
        # Internal epoch counter
        self.epoch_var = K.variable(0, dtype="int32", name="current_epoch")
        self.swap_epoch = swap_epoch

        # Two instances of your custom loss
        self.loss1 = CustomBinaryFocalLoss(**shared_kwargs, smoothing_as_correct=True)
        self.loss2 = CustomBinaryFocalLoss(**shared_kwargs, smoothing_as_correct=False)

    def call(self, y_true, y_pred):
        # Branch in graph
        return tf.cond(
            tf.less(self.epoch_var, self.swap_epoch),
            lambda: self.loss1(y_true, y_pred),
            lambda: self.loss2(y_true, y_pred),
        )

    def get_config(self):
        base = super().get_config()
        base.update({
            "swap_epoch": self.swap_epoch,
            # plus any other args you want to save…
        })
        return base
    
@utils.register_keras_serializable 
class SwitchingBinaryCrossentropyLoss(Loss):
    def __init__(self,
                 swap_epoch=3,
                 **shared_kwargs):
        super().__init__(name="switching_focal_loss")
        # Internal epoch counter
        self.epoch_var = K.variable(0, dtype="int32", name="current_epoch")
        self.swap_epoch = swap_epoch

        # Two instances of your custom loss
        self.loss1 = CustomBinaryCrossentropyLoss(**shared_kwargs, smoothing_as_correct=True)
        self.loss2 = CustomBinaryCrossentropyLoss(**shared_kwargs, smoothing_as_correct=False)

    def call(self, y_true, y_pred):
        # Branch in graph
        return tf.cond(
            tf.less(self.epoch_var, self.swap_epoch),
            lambda: self.loss1(y_true, y_pred),
            lambda: self.loss2(y_true, y_pred),
        )

    def get_config(self):
        base = super().get_config()
        base.update({
            "swap_epoch": self.swap_epoch,
            # plus any other args you want to save…
        })
        return base

@utils.register_keras_serializable
class EpochUpdater(tf.keras.callbacks.Callback):
    def __init__(self, loss_obj):
        super().__init__()
        self.loss_obj = loss_obj

    def on_epoch_end(self, epoch, logs=None):
        # epoch runs 0,1,2,… so set to epoch+1
        K.set_value(self.loss_obj.epoch_var, epoch + 1)
   
# Instantiate switchable loss:
# switchable_loss = SwitchingFocalLoss(
#     swap_epoch=3,
#     dominant_class_index=0,
#     focal_gamma=2.0,
#     focal_alpha=0.25,
#     # …etc…
# )

# model.compile(optimizer="adam", loss=switchable_loss, metrics=[…])

# # Fit with our callback that bumps the epoch counter inside the loss
# model.fit(
#     x_train, y_train,
#     epochs=10,
#     callbacks=[EpochUpdater(switchable_loss)],
#     …
# )