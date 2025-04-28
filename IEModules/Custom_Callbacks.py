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
import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Input, Model, layers, metrics, losses, callbacks, optimizers, models, utils
from keras import backend as K
import gc
import keras_tuner as kt
from pyfaidx import Fasta
import matplotlib.pyplot as plt

from config import DATA_DIR, LOG_DIR, MODEL_DIR, MODULE_DIR, NOTEBOOK_DIR
from config import (
    MODEL_DIR,
    CHECKPOINT_SUBDIR,
    CHECKPOINT_FILENAME,
    CHECKPOINT_MONITOR,
    CHECKPOINT_MODE,
    CHECKPOINT_SAVE_BEST_ONLY,
    CHECKPOINT_SAVE_WEIGHTS_ONLY,
    CHECKPOINT_SAVE_FREQ,
    LR_STATE_SAVE_SUBDIR,
    LR_STATE_SAVE_FILENAME,
    experiment_folder
)

class TimeLimit(callbacks.Callback):
    def __init__(self, max_time_seconds):
        super().__init__()
        self.max_time_seconds = max_time_seconds
        self.start_time = None

    def on_train_begin(self, logs=None):
        self.start_time = time.time()

    # def on_batch_end(self, batch, logs=None):
    #     if time.time() - self.start_time > self.max_time_seconds:
    #         self.model.stop_training = True
    
    # def on_train_batch_end(self, batch, logs=None):  # ✅ Runs more frequently than `on_batch_end`
    #     elapsed_time = time.time() - self.start_time
    #     if elapsed_time > self.max_time_seconds:
    #         print(f"\n⏳ Time limit of {self.max_time_seconds} sec reached. Stopping training!")
    #         self.model.stop_training = True  # 🔥 Stops training mid-batch
    
    def on_train_batch_begin(self, batch, logs=None):
        elapsed_time = time.time() - self.start_time
        if elapsed_time > self.max_time_seconds:
            print(f"\n⏳ Time limit of {self.max_time_seconds} sec reached. Stopping training!")
            self.model.stop_training = True

    def on_epoch_end(self, epoch, logs=None):  # New method added
        if time.time() - self.start_time > self.max_time_seconds:
            self.model.stop_training = True
            
class DebugCallback(callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print(f"\n🚀 Starting Epoch {epoch+1}")
        sys.stdout.flush()

    def on_batch_begin(self, batch, logs=None):
        if batch % 1000 == 0:
            print(f"🔄 Processing Batch {batch}")
            sys.stdout.flush()

    def on_batch_end(self, batch, logs=None):
        if batch % 1000 == 0:
            print(f"✅ Finished Batch {batch}")
            sys.stdout.flush()

    def on_epoch_end(self, epoch, logs=None):
        print(f"\n🏁 Epoch {epoch+1} Completed!")
        sys.stdout.flush()
        
# class CleanupCallback(callbacks.Callback):
#     def on_epoch_end(self, epoch, logs=None):
#         # Force garbage collection
#         gc.collect()
#         print(f"Cleanup done at the end of epoch {epoch+1}")
        

# checkpoint_cb = callbacks.ModelCheckpoint(
#     filepath=MODEL_DIR + 'checkpoints/epoch-{epoch:03d}-val_no_background_f1-{val_no_background_f1:.4f}.keras',
#     # monitor='val_loss',          # what metric to name file on
#     monitor='val_no_background_f1',
#     mode='max',                    # Required for monitoring f1, comment out if monitoring val loss
#     save_best_only=False,        # save model always 
#     save_weights_only=False,     # save full model (architecture + weights)
#     save_freq='epoch'
# )


# early_stopping_cb = callbacks.EarlyStopping(
#     # monitor='val_loss',
#     monitor='val_no_background_f1',
#     mode='max',
#     patience=20,
#     min_delta=1e-4,
#     restore_best_weights=True
# )

class CleanupCallback(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Example: force garbage collection
        gc.collect()
        
class SampleCountStopping(callbacks.Callback):
    def __init__(self, max_samples):
        """
        Args:
            max_samples (int): The total number of samples to process before stopping.
        """
        super(SampleCountStopping, self).__init__()
        self.max_samples = max_samples
        self.samples_processed = 0

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        # 'size' is the number of samples in the current batch.
        batch_size = logs.get('size', 0)
        self.samples_processed += batch_size

        # Optionally, print or log progress.
        if self.samples_processed % 1000 < batch_size:
            print(f"Samples processed so far: {self.samples_processed}")

        if self.samples_processed >= self.max_samples:
            print(f"Reached target of {self.max_samples} samples. Stopping training.")
            self.model.stop_training = True
            

class StatefulReduceLROnPlateau(callbacks.ReduceLROnPlateau):
    """
    A subclass of ReduceLROnPlateau that adds the ability
    to save and load its state. It automatically saves its state at the end of every epoch.
    """
    def __init__(self, *args, state_save_filepath=None, **kwargs):
        """
        Initialize the scheduler with an optional file path for state saving.
        
        Args:
            state_save_filepath (str): The file path where the state will be saved at the end of each epoch.
        """
        super(StatefulReduceLROnPlateau, self).__init__(*args, **kwargs)
        self.state_save_filepath = state_save_filepath

    def get_state(self):
        """
        Return a dictionary containing the scheduler state.
        """
        # Key state variables include the wait counter, cooldown counter, and best metric value.
        return {
            "wait": self.wait,
            "cooldown_counter": self.cooldown_counter,
            "best": self.best,
        }

    def set_state(self, state):
        """
        Restore the scheduler state from a dictionary.
        """
        self.wait = state.get("wait", 0)
        self.cooldown_counter = state.get("cooldown_counter", 0)
        if "best" in state:
            self.best = state["best"]
        else:
            self.best = float("inf") if self.mode == "min" else -float("inf")

    def save_state_to_file(self, filepath):
        """
        Save the scheduler state to a JSON file.
        """
        state = self.get_state()
        with open(filepath, 'w') as f:
            json.dump(state, f)
        print(f"ReduceLROnPlateau state saved to {filepath}")

    def load_state_from_file(self, filepath):
        """
        Load the scheduler state from a JSON file.
        """
        with open(filepath, 'r') as f:
            state = json.load(f)
        self.set_state(state)
        print(f"ReduceLROnPlateau state loaded from {filepath}")

    def on_epoch_end(self, epoch, logs=None):
        """
        Call parent on_epoch_end and then save state if a file path is provided.
        """
        super().on_epoch_end(epoch, logs)
        if self.state_save_filepath:
            self.save_state_to_file(self.state_save_filepath)

# ===== Example Usage =====

# Set up the custom callback with a path for auto-saving state.
# reduce_lr = StatefulReduceLROnPlateau(
#     monitor='val_no_background_f1',
#     factor=0.5,
#     patience=5,
#     min_lr=1e-6,
#     verbose=1,
#     state_save_filepath='reduce_lr_state.json'
# )

# Use the callback during training:
# model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=..., callbacks=[reduce_lr, ...])
cleanup_cb = CleanupCallback()

checkpoint_dir = experiment_folder / CHECKPOINT_SUBDIR
checkpoint_dir.mkdir(parents=True, exist_ok=True)

checkpoint_cb = callbacks.ModelCheckpoint(
    filepath=str(checkpoint_dir / CHECKPOINT_FILENAME),
    monitor=CHECKPOINT_MONITOR,
    mode=CHECKPOINT_MODE,
    save_best_only=CHECKPOINT_SAVE_BEST_ONLY,
    save_weights_only=CHECKPOINT_SAVE_WEIGHTS_ONLY,
    save_freq=CHECKPOINT_SAVE_FREQ,
)

early_stopping_cb = callbacks.EarlyStopping(
    monitor=CHECKPOINT_MONITOR,
    mode=CHECKPOINT_MODE,
    patience=20,
    min_delta=1e-4,
    restore_best_weights=True
)

lr_state_dir = experiment_folder / LR_STATE_SAVE_SUBDIR
lr_state_dir.mkdir(parents=True, exist_ok=True)
lr_state_path = lr_state_dir / LR_STATE_SAVE_FILENAME

reduce_lr_cb = StatefulReduceLROnPlateau(
    monitor=CHECKPOINT_MONITOR,
    mode=CHECKPOINT_MODE,
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1,
    state_save_filepath=lr_state_path
)

# Running this in the experiment so trial number is part of the name
# tb_log_dir = "./Logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_cb = callbacks.TensorBoard(log_dir=tb_log_dir, histogram_freq=1)

# Accessing tensorboard
# Bash:
# tensorboard --logdir=logs/fit
# Open a browser and go to http://localhost:6006

CALLBACKS = [
    cleanup_cb,
    checkpoint_cb,
    early_stopping_cb,
    reduce_lr_cb,
    ]