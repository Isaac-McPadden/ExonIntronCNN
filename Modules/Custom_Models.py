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
def tile_to_batch(z):
    pe, x = z
    return tf.tile(pe, [tf.shape(x)[0], 1, 1])

@utils.register_keras_serializable()
def create_dcnn_model(
    input_dim=5,
    sequence_length=5000,
    num_classes=5
):
    inputs = Input(shape=(sequence_length, input_dim))
    
    # Condensed positional encoding block.  See cnn for description
    positions = tf.range(start=0, limit=sequence_length, delta=1)
    pos_encoding = layers.Embedding(input_dim=sequence_length, output_dim=num_classes)(positions)
    pos_encoding = tf.expand_dims(pos_encoding, axis=0)
    # def tile_to_batch(z):
    #     pe, x = z
    #     return tf.tile(pe, [tf.shape(x)[0], 1, 1])
    pos_encoding = layers.Lambda(tile_to_batch)([pos_encoding, inputs])

    concat_input = layers.Concatenate(axis=-1)([inputs, pos_encoding])
    
    '''Initial training hyperparameters'''
    # early_dropout = 0
    # middle_dropout = 0.1
    # late_dropout = 0.2
    
    '''Reducing false positive distance hyperparameters'''
    early_dropout = 0.1
    middle_dropout = 0
    late_dropout = 0.2

    cnn = layers.Conv1D(filters=64, kernel_size=9, activation='relu', padding='same')(concat_input)
    cnn = layers.BatchNormalization()(cnn)
    cnn = layers.Dropout(early_dropout)(cnn)
    # We use six layers with increasing dilation rates to capture a wider receptive field.
    # Dilating convolutional blocks with dropout (pooling is bad because exact sequence matters)
    skip = concat_input
    skip = layers.Conv1D(filters=64, kernel_size=1, padding='same')(skip)
    dcnn = layers.Conv1D(filters=64, kernel_size=9, dilation_rate=1, activation='relu', padding='same')(skip)
    dcnn = layers.BatchNormalization()(dcnn)
    dcnn = layers.Dropout(early_dropout)(dcnn)
    low_dcnn = dcnn
    
    dcnn = layers.Conv1D(filters=64, kernel_size=9, dilation_rate=2, activation='relu', padding='same')(dcnn)
    dcnn = layers.BatchNormalization()(dcnn)
    dcnn = layers.Dropout(early_dropout)(dcnn)
    dcnn = layers.Add()([dcnn, skip])
    
    skip = dcnn
    skip = layers.Conv1D(filters=160, kernel_size=1, padding='same')(skip)
    dcnn = layers.Conv1D(filters=160, kernel_size=9, dilation_rate=4, activation='relu', padding='same')(dcnn)
    dcnn = layers.BatchNormalization()(dcnn)
    dcnn = layers.Dropout(middle_dropout)(dcnn)
    
    dcnn = layers.Conv1D(filters=160, kernel_size=9, dilation_rate=8, activation='relu', padding='same')(dcnn)
    dcnn = layers.BatchNormalization()(dcnn)
    dcnn = layers.Dropout(middle_dropout)(dcnn)
    dcnn = layers.Add()([dcnn, skip])
    
    skip = dcnn
    skip = layers.Conv1D(filters=192, kernel_size=1, padding='same')(skip)
    dcnn = layers.Conv1D(filters=192, kernel_size=9, dilation_rate=16, activation='relu', padding='same')(dcnn)
    dcnn = layers.BatchNormalization()(dcnn)
    dcnn = layers.Dropout(middle_dropout)(dcnn)
    
    dcnn = layers.Conv1D(filters=192, kernel_size=9, dilation_rate=32, activation='relu', padding='same')(dcnn)
    dcnn = layers.BatchNormalization()(dcnn)
    dcnn = layers.Dropout(middle_dropout)(dcnn)
    dcnn = layers.Add()([dcnn, skip])
    
    skip = dcnn
    skip = layers.Conv1D(filters=192, kernel_size=1, padding='same')(skip)
    dcnn = layers.Conv1D(filters=192, kernel_size=9, dilation_rate=64, activation='relu', padding='same')(dcnn)
    dcnn = layers.BatchNormalization()(dcnn)
    dcnn = layers.Dropout(middle_dropout)(dcnn)
    
    dcnn = layers.Conv1D(filters=192, kernel_size=9, dilation_rate=128, activation='relu', padding='same')(dcnn)
    dcnn = layers.BatchNormalization()(dcnn)
    dcnn = layers.Dropout(middle_dropout)(dcnn)
    dcnn = layers.Add()([dcnn, skip])
    
    # Adding more layers here and increasing dense filters to 512 did not improve results
    # skip = dcnn
    # skip = layers.Conv1D(filters=256, kernel_size=1, padding='same')(skip)
    # dcnn = layers.Conv1D(filters=256, kernel_size=9, dilation_rate=256, activation='relu', padding='same')(dcnn)
    # dcnn = layers.BatchNormalization()(dcnn)
    # dcnn = layers.Dropout(middle_dropout)(dcnn)
    
    # dcnn = layers.Conv1D(filters=256, kernel_size=9, dilation_rate=512, activation='relu', padding='same')(dcnn)
    # dcnn = layers.BatchNormalization()(dcnn)
    # dcnn = layers.Dropout(middle_dropout)(dcnn)
    # dcnn = layers.Add()([dcnn, skip])
    
    # skip = dcnn
    # skip = layers.Conv1D(filters=256, kernel_size=1, padding='same')(skip)
    # dcnn = layers.Conv1D(filters=256, kernel_size=9, dilation_rate=1024, activation='relu', padding='same')(dcnn)
    # dcnn = layers.BatchNormalization()(dcnn)
    # dcnn = layers.Dropout(middle_dropout)(dcnn)
    
    # dcnn = layers.Conv1D(filters=256, kernel_size=9, dilation_rate=2048, activation='relu', padding='same')(dcnn)
    # dcnn = layers.BatchNormalization()(dcnn)
    # dcnn = layers.Dropout(middle_dropout)(dcnn)
    # dcnn = layers.Add()([dcnn, skip])
    
    second_concat = layers.Concatenate(axis=-1)([concat_input, cnn, dcnn, low_dcnn])

    # Instead of flattening, use Conv1D with kernel_size=1 as dense layers:
    dense = layers.Conv1D(128, kernel_size=1, activation='relu')(second_concat)
    dense = layers.BatchNormalization()(dense)
    dense = layers.Dropout(late_dropout)(dense)
    
    dense = layers.Conv1D(128, kernel_size=1, activation='relu')(dense)
    dense = layers.BatchNormalization()(dense)
    dense = layers.Dropout(late_dropout)(dense)

    # Final classification layer applied at every time step:
    outputs = layers.Conv1D(num_classes, kernel_size=1, activation='sigmoid')(dense)

    model = Model(inputs=inputs, outputs=outputs)
    return model