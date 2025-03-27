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

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Input, Model, layers, metrics, losses, callbacks, optimizers, models, utils
from keras import backend as K
import gc
import keras_tuner as kt
from pyfaidx import Fasta
import matplotlib.pyplot as plt

datasets_path = "../../Datasets/"
models_path = "../../Models/"

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
        
class CleanupCallback(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Force garbage collection
        gc.collect()
        print(f"Cleanup done at the end of epoch {epoch+1}")
        

checkpoint_cb = callbacks.ModelCheckpoint(
    filepath=models_path + 'checkpoints/epoch-{epoch:03d}-val_no_background_f1-{val_no_background_f1:.4f}.keras',
    # monitor='val_loss',          # what metric to name file on
    monitor='val_no_background_f1',
    mode='max',                    # Required for monitoring f1, comment out if monitoring val loss
    save_best_only=False,        # save model always 
    save_weights_only=False,     # save full model (architecture + weights)
    save_freq='epoch'
)


early_stopping_cb = callbacks.EarlyStopping(
    # monitor='val_loss',
    monitor='val_no_background_f1',
    mode='max',
    patience=20,
    min_delta=1e-4,
    restore_best_weights=True
)

class CleanupCallback(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Example: force garbage collection
        gc.collect()