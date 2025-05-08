from __future__ import annotations

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
from pathlib import Path
import tempfile
import shutil
import errno

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
    PREVAL_CHECKPOINT_MONITOR,
    PREVAL_CHECKPOINT_FILENAME,
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
        self._cur_epoch = epoch

    def on_train_batch_end(self, batch, logs=None):
        if batch + 1 == self.steps_per_epoch:
            # Keras 3 signature: epoch, batch, logs
            self._save_model(epoch=self._cur_epoch,
                             batch=batch,
                             logs=logs or {})
    def _save_model(self, epoch, batch, logs=None):
        super()._save_model(epoch, batch, logs)


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
            lr_var = getattr(self.model.optimizer, "lr",
                              getattr(self.model.optimizer, "learning_rate", None))
            if lr_var is not None:
                lr = float(tf.convert_to_tensor(lr_var).numpy())
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
            lr_var = getattr(self.model.optimizer, "lr",
                             getattr(self.model.optimizer, "learning_rate", None))
            if lr_var is not None:
                lr_var.assign(self._saved_lr)

    def on_train_begin(self, logs=None):
        """Guard-rail in case `set_model` wasn't enough (very rare)."""
        if self._saved_lr is not None:
            lr_var = getattr(self.model.optimizer, "lr",
                             getattr(self.model.optimizer, "learning_rate", None))
            if lr_var is not None:
                lr_var.assign(self._saved_lr)
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


@utils.register_keras_serializable()
class EpochSetter(callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        # this var now lives on GPU, so assign() is safe
        self.model.loss.current_epoch.assign(epoch)
        
    def get_config(self):
        return {}
    
    
# ►► 1. helper – create a *closed* temporary file *next* to the final file
def _atomic_temp(target: Path) -> Path:
    """
    Return a Path to an empty temporary file living in `target.parent`.
    The file descriptor is closed immediately so it's renamable on Windows.
    """
    fd, tmp = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.stem}_partial_",
        suffix=target.suffix
    )
    os.close(fd)
    return Path(tmp)


# ►► 2. mix‑in – transparently wrap `_save_model` in “write‑then‑rename”
class _AtomicMixin:
    """
    Add atomic‑write semantics to any Keras *Checkpoint callback.

    Works with every current Keras signature because we forward *args/**kwargs
    verbatim to the parent implementation.
    """

    def _save_model(self, *args, **kwargs):                      # pylint: disable=arguments-differ
        final_path = Path(self.filepath)
        tmp_path   = _atomic_temp(final_path)

        original_fp, self.filepath = self.filepath, str(tmp_path)
        try:
            super()._save_model(*args, **kwargs)                 # 1️⃣ write to temp
            for attempt in range(5):                             # 2️⃣ robust replace
                try:
                    tmp_path.replace(final_path)                 # atomic on POSIX+NTFS
                    break                                        # success
                except PermissionError as e:
                    if attempt == 4:                             # give up after 5 tries
                        raise
                    time.sleep(0.2)                              # brief back‑off
        finally:
            self.filepath = original_fp
            tmp_path.unlink(missing_ok=True)                     # clean up stray temp


# ►► 3. concrete classes to use in your callback list
class AtomicModelCheckpoint(_AtomicMixin, callbacks.ModelCheckpoint):
    """
    Ordinary *post‑validation* checkpoint that is written atomically.
    Inherit exactly the same constructor / behaviour as keras.callbacks.ModelCheckpoint.
    """
    pass


class AtomicBatchModelCheckpoint(_AtomicMixin, callbacks.ModelCheckpoint):
    """
    Save exactly once per epoch – after the final training batch, before
    validation – with atomic write‑to‑temp behaviour.
    """
    def __init__(self, filepath: str | Path, *, steps_per_epoch: int, **kwargs):
        # keep Keras happy: save_freq must be 'epoch' or int → we pick 'epoch'
        super().__init__(filepath=str(filepath),
                         save_freq="epoch",
                         **kwargs)
        self.steps_per_epoch = steps_per_epoch
        self._cur_epoch = None

    # track current epoch
    def on_epoch_begin(self, epoch, logs=None):
        self._cur_epoch = epoch

    # fire once at the last train batch
    def on_train_batch_end(self, batch, logs=None):
        if (batch + 1) == self.steps_per_epoch:
            self._save_model(epoch=self._cur_epoch, batch=batch, logs=logs)

    # silence the normal post‑epoch save to avoid duplicates
    def on_epoch_end(self, epoch, logs=None):
        pass


@utils.register_keras_serializable()
class TidyCheckpointNames(callbacks.Callback):
    def __init__(self, ckpt_dir, **kwargs):
        super().__init__(**kwargs)
        self.ckpt_dir = Path(ckpt_dir)

    _junk = re.compile(
        r"""
        ^\.                                   |   # leading dot
        (?:_partial)?_[0-9a-z_]{6,15}(?=\.keras$)    # trailing slug
        """,
        re.X,
)

    def _clean(self, name: str) -> str:
        return self._junk.sub("", name)

    def on_epoch_end(self, epoch, logs=None):
        for p in self.ckpt_dir.glob("*.keras"):
            clean = self._clean(p.name)
            if clean != p.name:
                target = p.with_name(clean)
                if not target.exists():          # don’t clobber            '
                    p.rename(target)
                    
    def get_config(self):
        cfg = super().get_config()
        cfg.update({"ckpt_dir": str(self.ckpt_dir)})  # convert Path → str
        return cfg

    @classmethod
    def from_config(cls, config):
        # turn the string back into a Path object
        config = dict(config)
        config["ckpt_dir"] = Path(config["ckpt_dir"])
        return cls(**config)

# Use the callback during training:
# model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=..., callbacks=[reduce_lr, ...])
cleanup_cb = CleanupCallback()

checkpoint_dir = experiment_folder / CHECKPOINT_SUBDIR
checkpoint_dir.mkdir(parents=True, exist_ok=True)

checkpoint_cb = AtomicModelCheckpoint(
    filepath=str(checkpoint_dir / CHECKPOINT_FILENAME),
    monitor=CHECKPOINT_MONITOR,
    mode=CHECKPOINT_MODE,
    save_best_only=CHECKPOINT_SAVE_BEST_ONLY,
    save_weights_only=CHECKPOINT_SAVE_WEIGHTS_ONLY,
    save_freq=CHECKPOINT_SAVE_FREQ,
)

batch_checkpoint_cb = AtomicBatchModelCheckpoint(
    filepath=str(checkpoint_dir / PREVAL_CHECKPOINT_FILENAME),
    steps_per_epoch=STEPS_PER_EPOCH_UNIT,
    monitor=PREVAL_CHECKPOINT_MONITOR,
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

tidy_cb = TidyCheckpointNames(checkpoint_dir)
# Running this in the experiment so trial number is part of the name
# tb_log_dir = "./Logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_cb = callbacks.TensorBoard(log_dir=tb_log_dir, histogram_freq=1)

# Accessing tensorboard
# Bash:
# tensorboard --logdir=logs/fit
# Open a browser and go to http://localhost:6006

CALLBACKS = [
    batch_checkpoint_cb,
    cleanup_cb,
    checkpoint_cb,
    early_stopping_cb,
    reduce_lr_cb,
    tidy_cb,
    ]