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

from IEModules.config import DATA_DIR, LOG_DIR, MODEL_DIR, MODULE_DIR, NOTEBOOK_DIR
from IEModules.config import (
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
    experiment_folder,
    STEPS_PER_EPOCH_UNIT,
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


class BatchModelCheckpoint(callbacks.ModelCheckpoint):
    """
    A ModelCheckpoint that also saves immediately after the final training
    batch of each epoch (i.e. before validation begins), using the same
    filepath template and arguments as the standard checkpoint.
    """
    def __init__(self, filepath, steps_per_epoch, logs=None, **kwargs):
        """
        Args:
            filepath: same template you’d pass to ModelCheckpoint
                      (e.g. ".../epoch-{epoch:03d}.keras")
            steps_per_epoch: number of train batches per epoch
            **kwargs: all the same keyword args you’d pass to ModelCheckpoint
        """
        super().__init__(filepath, **kwargs)
        self.steps_per_epoch = steps_per_epoch
        self._current_epoch = None

    def on_epoch_begin(self, epoch, logs=None):
        # track which epoch we’re in for later filename formatting
        self._current_epoch = epoch

    def on_train_batch_end(self, batch, logs=None):
        # zero‐based batch index → check for last batch
        if batch + 1 == self.steps_per_epoch:
            # use the built‐in saving logic, passing the tracked epoch
            self._save_model(self._current_epoch, logs=None)        


class StatefulReduceLROnPlateau(callbacks.ReduceLROnPlateau):
    """
    ReduceLROnPlateau that can be resumed across runs.
    Saves / restores wait-counter, cooldown, best metric **and** the
    exact optimizer learning-rate.
    """
    def __init__(self, *args, state_save_filepath=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.state_save_filepath = state_save_filepath
        self._saved_lr = None          #  <-- populated by `set_state`

    # ------------------------------------------------------------------ #
    #                      STATE SERIALISATION                           #
    # ------------------------------------------------------------------ #
    def get_state(self):
        """Return a dict that fully recreates the scheduler state."""
        lr = None
        if self.model is not None:                     # model set after build()
            lr = float(K.get_value(self.model.optimizer.lr))
        return {
            "wait":             self.wait,
            "cooldown_counter": self.cooldown_counter,
            "best":             self.best,
            "lr":               lr,
        }

    def set_state(self, state):
        """Restore internal counters **and** remember the LR to re-inject later."""
        self.wait             = state.get("wait", 0)
        self.cooldown_counter = state.get("cooldown_counter", 0)
        self.best             = state.get(
            "best",
            float("inf") if self.mode == "min" else -float("inf")
        )
        self._saved_lr = state.get("lr", None)

    # ------------------------------------------------------------------ #
    #               HOOKS THAT RE-APPLY THE SAVED LR                     #
    # ------------------------------------------------------------------ #
    def set_model(self, model):
        """Called by Keras just before training starts."""
        super().set_model(model)
        if self._saved_lr is not None:
            try:                                    # Keras ≥2.13 uses `.lr`
                K.set_value(self.model.optimizer.lr, self._saved_lr)
            except AttributeError:                  # if using `.learning_rate`
                K.set_value(self.model.optimizer.learning_rate, self._saved_lr)

    def on_train_begin(self, logs=None):
        """Guard-rail in case `set_model` wasn't enough (very rare)."""
        if self._saved_lr is not None:
            try:
                K.set_value(self.model.optimizer.lr, self._saved_lr)
            except AttributeError:
                K.set_value(self.model.optimizer.learning_rate, self._saved_lr)
        super().on_train_begin(logs)

    # ------------------------------------------------------------------ #
    #                     SAVE–ON-TRAIN-AND-EPOCH-END                              #
    # ------------------------------------------------------------------ #
    def save_state_to_file(self, filepath):
        with open(filepath, "w") as f:
            json.dump(self.get_state(), f)
        print(f"ReduceLROnPlateau state saved to {filepath}")

    def load_state_from_file(self, filepath):
        with open(filepath, "r") as f:
            state = json.load(f)
        self.set_state(state)
        print(f"ReduceLROnPlateau state loaded from {filepath}")

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        if self.state_save_filepath:
            self.save_state_to_file(self.state_save_filepath)
    
    def on_train_end(self, logs=None):
        """
        Called after the final training batch of every epoch but *before*
        validation starts.  Perfect place to persist LR-scheduler state so
        killing the job during validation costs you nothing.
        """
        super().on_train_end(logs)
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

batch_checkpoint_cb = BatchModelCheckpoint(
    filepath=str(checkpoint_dir / CHECKPOINT_FILENAME),
    steps_per_epoch=STEPS_PER_EPOCH_UNIT,
    monitor=CHECKPOINT_MONITOR,
    mode=CHECKPOINT_MODE,
    save_best_only=CHECKPOINT_SAVE_BEST_ONLY,
    save_weights_only=CHECKPOINT_SAVE_WEIGHTS_ONLY,
    verbose = 1,
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
    batch_checkpoint_cb,
    ]