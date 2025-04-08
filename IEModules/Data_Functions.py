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

datasets_path = "../Datasets/"
models_path = "../Models/"


def drop_exact_records(dataset: tf.data.Dataset, total_records, num_to_drop, seed=None):
    '''
    Function to drop n records from data before constructing parsed dataset.  
    Mostly for bug checking.
    '''
    if seed:
        np.random.seed(seed)
    drop_indices = set(np.random.choice(total_records, num_to_drop, replace=False))
    dataset = dataset.enumerate()
    dataset = dataset.filter(lambda i, x: ~tf.reduce_any(tf.equal(i, list(drop_indices))))
    dataset = dataset.map(lambda i, x: x)
    return dataset


def parse_chunk_example(serialized_example):
    """
    Parses a single serialized tf.train.Example back into tensors.
    Used in testing datasets and in piping tfrecords to DL Algorithms
    """
    feature_spec = {
        'X':          tf.io.VarLenFeature(tf.float32),
        'y':          tf.io.VarLenFeature(tf.float32),
        'record_id':  tf.io.FixedLenFeature([], tf.string),
        'cstart':     tf.io.FixedLenFeature([1], tf.int64),
        'cend':       tf.io.FixedLenFeature([1], tf.int64),
        'strand':     tf.io.FixedLenFeature([], tf.string),
        'chunk_size': tf.io.FixedLenFeature([1], tf.int64),
    }
    
    parsed = tf.io.parse_single_example(serialized_example, feature_spec)
    
    # chunk_size is shape [1]
    chunk_size = parsed['chunk_size'][0]
    
    # Convert sparse to dense
    X_flat = tf.sparse.to_dense(parsed['X'])
    y_flat = tf.sparse.to_dense(parsed['y'])

    # Reshape X to [chunk_size, 5]
    X_reshaped = tf.reshape(X_flat, [chunk_size, 5])
    # Reshape y to [chunk_size], probably redundant
    y_reshaped = tf.reshape(y_flat, [chunk_size, 5])
    
    record_id = parsed['record_id']
    cstart = parsed['cstart'][0]
    cend = parsed['cend'][0]
    strand = parsed['strand']
    
    return X_reshaped, y_reshaped, record_id, cstart, cend, strand


def prepare_for_model(X, y, record_id, cstart, cend, strand, cut_background=False):
    '''
    Helper function that extracts and reshapes parsed data for feeding to DL Models
    '''
    if cut_background:
        # Remove the first column to get shape [chunk_size, 4]
        y = y[:, 1:]
    return X, y


def prep_dataset_from_tfrecord(
    tfrecord_path,
    batch_size=28,
    compression_type='GZIP',
    shuffled = False,
    shuffle_buffer=25000,
    total_records=None,
    num_to_drop=None,
    seed=None,
    cut_background=False
):
    '''
    Imports tfrecord and shuffles it then parses it for use in fitting a model
    
    :tfrecord_path: string giving path to tfrecord to load
    :batch_size: int giving batch size fed to model before calculating loss and metrics
    :compression_type: string, compression type of tfrecord to load, default is 'GZIP'
    :shuffled: Should tfrecord examples be shuffled before feeding to the model, default is False
    :shuffle_buffer: int, how many examples to hold in the shuffle pool, default is 25000
    :total_records: int, used with num_to_drop, default is None,
    :num_to_drop: int, used with total_records if you want to ensure the final batch default is None,
    :seed: int, seed for RNG, useful for ensuring val and test data are reconstructed identically every time, default is None
    :cut_background: bool, removes background column from labels, default is False
    
    Returns: a tf.data.Dataset
    '''
    # Loads in records in a round robin fashion for slightly increased mixing
    dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type=compression_type, num_parallel_reads = tf.data.AUTOTUNE)
    
    if num_to_drop:
        dataset = drop_exact_records(dataset, total_records=total_records, num_to_drop=num_to_drop, seed=seed)
    
    if shuffled == True:
        # Shuffle at the record level
        dataset = dataset.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
        
    
    dataset = dataset.map(parse_chunk_example, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(
        lambda X, y, record_id, cstart, cend, strand: prepare_for_model(X, y, record_id, cstart, cend, strand, cut_background=cut_background),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Rebatch parsed and prefetch for efficient reading
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset