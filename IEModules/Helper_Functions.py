import time
import sys
import os
import glob
import math
import threading
import concurrent.futures as cf
import random
import re
import json
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Input, Model, layers, metrics, losses, callbacks, optimizers, models, utils
from keras import backend as K
import gc
import keras_tuner as kt
from pyfaidx import Fasta
import matplotlib.pyplot as plt

from IEModules.config import DATA_DIR, LOG_DIR, MODEL_DIR, MODULE_DIR, NOTEBOOK_DIR

# Various helper functions related to trial data recording.

# The kind of stuff that before now, I was copy pasting from one notebook to the next

def plot_train_val_curve(history_object, training_target_variable: str):
    '''
    Plots the train-validation curve from a keras history object.  
    training_target_variable will be the name of one of the metrics being tracked
    '''
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(history_object.history[f'{training_target_variable}'], label=f'Training {training_target_variable}')
    ax.plot(history_object.history[f'val_{training_target_variable}'], label=f'Validation {training_target_variable}')
    ax.set_title(f'Train vs. Validation {training_target_variable}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(training_target_variable)
    ax.legend()
    plt.show()
    return fig 


# def save_history_to_json(history, metadata: str):
#     """
#     Saves the training history from a Keras model to a JSON file.
    
#     Parameters:
#         history: The History object returned by model.fit().
#         metadata (str): A string to prefix the filename for context.
#         Basically, once I've figured out a serial number system, it will be that
#     Returns:
#         str: The filename of the saved JSON file.
#     """
#     filename = f'{metadata}_training_history.json'
#     with open(filename, 'w') as f:
#         json.dump(history.history, f)
#         print(f'Saved {filename}')
#     return filename

def save_history_to_json(history, metadata: str, save_path: str = ""):
    """
    Saves the training history from a Keras model to a JSON file.

    Parameters:
        history: The History object returned by model.fit().
        metadata (str): A string used as a prefix for the filename.
                        Basically, once I've figured out a serial number system, it will be that
        save_path (str, optional): The directory where the JSON file will be saved. 
                                   Defaults to the current working directory.
    
    Returns:
        str: The complete file path of the saved JSON file.
    """
    if save_path:
        # Ensures the directory exists
        os.makedirs(save_path, exist_ok=True)
        filename = os.path.join(save_path, f'{metadata}_training_history.json')
    else:
        filename = f'{metadata}_training_history.json'
    
    with open(filename, 'w') as f:
        json.dump(history.history, f)
    print(f'Saved {filename}')
    return filename


def load_history_from_json(metadata: str, save_path: str = ""):
    """
    Loads the training history from a JSON file.

    Parameters:
        metadata (str): The prefix used for the filename when the history was saved.
        save_path (str, optional): The directory where the JSON file is saved.
                                   Defaults to the current working directory.

    Returns:
        dict: The loaded history dictionary.
    """
    if save_path:
        filename = os.path.join(save_path, f'{metadata}_training_history.json')
    else:
        filename = f'{metadata}_training_history.json'
    
    with open(filename, 'r') as f:
        history_dict = json.load(f)
    
    return history_dict


def fabricate_vectors_for_f1(target_f1: float,
                             length: int = 200,
                             include_background: bool = True,
                             focus_class: int = 1,
                             seed: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return y_true / y_pred matrices (shape = [length, num_classes]) whose
    **weighted** CustomNoBackgroundF1Score equals `target_f1` exactly.
    See full doc-string in previous message.
    """
    # -------------  parameter checks -------------
    if not (0 < target_f1 <= 1):
        raise ValueError("target_f1 must be in (0, 1].")
    if focus_class not in range(1, 5):
        raise ValueError("focus_class must be 1…4.")

    rng         = np.random.default_rng(seed)
    n_classes   = 5 if include_background else 4
    tgt_col_idx = focus_class if include_background else focus_class-1

    # ----------  pick a confusion‑matrix  ----------
    for TP in range(1, length):
        # Choose FN = 0   →   FP = (2·TP/F1) − 2·TP
        FP = (2 * TP / target_f1) - 2 * TP
        if FP.is_integer() and FP >= 0 and TP + FP <= length:
            FP, FN = int(FP), 0
            TN     = length - (TP + FP)
            break
    else:
        raise RuntimeError("Increase `length` – cannot realise that F‑score.")

    # ----------  materialise the rows  ----------
    y_true = np.zeros((length, n_classes), dtype=np.float32)
    y_pred = np.zeros_like(y_true)
    rows   = rng.permutation(length)
    k = 0
    for _ in range(TP):
        r = rows[k]; k += 1
        y_true[r, tgt_col_idx] = 1
        y_pred[r, tgt_col_idx] = 1
    for _ in range(FP):
        r = rows[k]; k += 1
        if include_background:
            y_true[r, 0] = 1
        y_pred[r, tgt_col_idx] = 1
    for _ in range(TN):
        r = rows[k]; k += 1
        if include_background:
            y_true[r, 0] = 1
    return y_true, y_pred